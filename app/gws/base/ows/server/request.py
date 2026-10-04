"""OWS service request."""

from typing import Optional, Callable, cast
import re

import gws
import gws.base.layer.core
import gws.base.legend
import gws.base.model
import gws.base.web
import gws.lib.extent
import gws.gis.render
import gws.lib.mime
import gws.lib.bounds
import gws.lib.crs
import gws.lib.image
import gws.lib.uom
import gws.lib.xmlx

from . import core, layer_caps, error


class TemplateArgs(gws.TemplateArgs):
    """Arguments for service templates."""

    featureCollection: core.FeatureCollection
    """Search result, for feature requests."""
    metadataCollection: core.MetadataCollection
    """Search result, for catalog requests."""
    operation: gws.OwsOperation
    """Requested operation."""
    project: gws.Project
    """Project."""
    request: 'Object'
    """Service request."""
    layerCapsList: list[core.LayerCaps]
    """Layer caps to include in the response."""
    serviceRequest: 'Object'
    """Service request."""
    service: gws.OwsService
    """Service object."""
    serviceUrl: str
    """Canonical service URL."""
    url_for: Callable
    """Function that converts a URL or path to a canonical URL."""
    gmlVersion: int
    """GML version for geometries."""
    version: str
    """Requested service version."""
    intVersion: int
    """Requested service version as an integer, e.g. ``130`` for ``1.3.0``."""
    tileMatrixSets: list[gws.TileMatrixSet]
    """Tile matrix sets (WMTS)."""


class Object:
    """OWS service request.

    Holds the request parameters and provides methods to read and validate them.
    The constructor determines the operation and the version; the service
    handlers fill in the other attributes (CRS, bounds, sizes) as needed.
    """

    alwaysXY: bool
    """Use XY axis order for all CRSs."""
    bounds: gws.Bounds
    """Requested bounds."""
    crs: gws.Crs
    """Requested CRS."""
    params: dict
    """Request parameters, with upper-cased names."""
    pxSize: gws.Size
    """Requested image size in pixels."""
    resolution: float
    """Requested resolution."""
    resX: float
    """Requested horizontal resolution."""
    resY: float
    """Requested vertical resolution."""
    isSoap: bool = False
    """The request was sent as a SOAP envelope."""
    layerCapsList: list[core.LayerCaps]
    """Caps of all layers available to the user in this service and project."""
    operation: gws.OwsOperation
    """Requested operation."""
    project: gws.Project
    """Project."""
    req: gws.WebRequester
    """Web request."""
    service: gws.OwsService
    """Service object."""
    targetCrs: gws.Crs
    """CRS of the output."""
    version: str
    """Negotiated service version."""
    xmlElement: Optional[gws.XmlElement]
    """Request body for XML POST requests."""
    customNamespacePrefixes: dict
    """Custom namespace prefixes (uri -> prefix) requested with the ``NAMESPACES`` parameter."""

    def __init__(
        self,
        service: gws.OwsService,
        req: gws.WebRequester,
        params: dict,
        xml_element: gws.XmlElement = None,
        is_soap=False,
    ) -> None:
        """Create a service request.

        Args:
            service: Service object.
            req: Web request.
            params: Request parameters.
            xml_element: Request body for XML POST requests.
            is_soap: The request was sent as a SOAP envelope.

        Raises:
            ``error.OperationNotSupported``: If the operation is not supported.
            ``error.VersionNegotiationFailed``: If none of the requested versions is supported.
            ``error.CurrentUpdateSequence``: If the requested update sequence equals the current one.
            ``error.InvalidUpdateSequence``: If the requested update sequence is greater than the current one.
        """
        self.service = service
        self.req = req
        self.project = cast(gws.Project, None)
        self.params = gws.u.to_upper_dict(params)
        self.xmlElement = xml_element
        self.isSoap = is_soap

        self.operation = self.requested_operation('REQUEST')
        self.version = self.requested_version('VERSION,ACCEPTVERSIONS')

        self.alwaysXY = False
        self.pxSize = 0, 0
        self.resolution = 0
        self.resX = 0
        self.resY = 0

        # OGC 06-042, 7.2.3.5
        if self.service.updateSequence:
            s = self.string_param('UPDATESEQUENCE', default='')
            if s and s == self.service.updateSequence:
                raise error.CurrentUpdateSequence()
            if s and s > self.service.updateSequence:
                raise error.InvalidUpdateSequence()

        self.customNamespacePrefixes = self.requested_xmlns_replacements()

    def require_project(self):
        """Load the project and the layer caps, the project is required.

        Raises:
            ``gws.NotFoundError``: If the project is not found.
            ``gws.ForbiddenError``: If the user cannot access the project.
        """

        return self.load_project(required=True)

    def load_project(self, required=False):
        """Load the project and the layer caps.

        The project is the one given by the ``projectUid`` parameter or the
        project the service is configured for. Sets ``project`` and
        ``layerCapsList``. The layer caps are cached per service, project and user roles.

        Args:
            required: Raise an error if there is no project.

        Raises:
            ``gws.NotFoundError``: If the project is required and not found, or does not match the service project.
            ``gws.ForbiddenError``: If the user cannot access the project.
        """

        # services can be configured globally (in which case, service.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (service.project != None)

        p = self.req.param('projectUid')
        project = None

        if p:
            project = self.req.user.require_project(p)
            if self.service.project and project != self.service.project:
                raise gws.NotFoundError(f'ows {self.service.uid}: wrong project={p!r}')
        elif self.service.project:
            # for in-project services, ensure the user can access the project
            project = self.req.user.require_project(self.service.project.uid)

        if not project:
            if required:
                raise gws.NotFoundError(f'ows {self.service.uid}: project not found')
            return

        self.project = project
        cache_key = 'layer_caps_' + gws.u.sha256([self.service.uid, self.project.uid, sorted(self.req.user.roles)])
        self.layerCapsList = gws.u.get_app_global(cache_key, self.enum_layer_caps)

    def enum_layer_caps(self):
        """Create the layer caps for the service root layer or the project root layer.

        Only layers the user can read and which are enabled for this service are
        included. Empty groups are skipped. Groups are listed before their children.

        Returns:
            A flat list of layer caps.
        """

        lcs = []
        root_layer = self.service.rootLayer or self.project.map.rootLayer
        self._enum_layer_caps(root_layer, lcs, [])
        return lcs

    def _enum_layer_caps(self, layer: gws.Layer, lcs: list[core.LayerCaps], stack: list[core.LayerCaps]):
        """Add caps for a layer and its sub-layers to ``lcs``, linking them to the groups in ``stack``."""
        if not self.req.user.can_read(layer) or not layer.isEnabledForOws:
            return
        
        ows = layer.ows
        if ows and ows.allowedServiceUids and self.service.uid not in ows.allowedServiceUids:
            return
        if ows and ows.deniedServiceUids and self.service.uid in ows.deniedServiceUids:
            return

        is_compat = self.service.layer_is_compatible(layer)
        if not is_compat and not layer.isGroup:
            return

        lc = layer_caps.for_layer(layer, self.req.user, self.service)

        # NB groups must be inspected even if not 'compatible'
        if layer.isGroup:
            lc.isGroup = True
            n = len(lcs)
            for sub_layer in layer.layers:
                self._enum_layer_caps(sub_layer, lcs, stack + [lc])
            if not lc.children:
                # no empty groups
                return
            if is_compat:
                lc.hasLegend = any(c.hasLegend for c in lc.children)
                lc.isSearchable = any(c.isSearchable for c in lc.children)
                lcs.insert(n, lc)
        else:
            lc.isGroup = False
            lcs.append(lc)
            for sup_lc in stack:
                sup_lc.leaves.append(lc)

        if stack:
            stack[-1].children.append(lc)

    ##

    def requested_version(self, param_names: str) -> str:
        """Negotiate the service version.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The first supported version that starts with a requested version,
            or the first supported version if no version is requested.

        Raises:
            ``error.VersionNegotiationFailed``: If none of the requested versions is supported.
        """

        p, val = self._get_param(param_names, '')
        if not val:
            # the first supported version is the default
            return self.service.supportedVersions[0]

        for v in gws.u.to_list(val):
            for ver in self.service.supportedVersions:
                if ver.startswith(v):
                    return ver

        raise error.VersionNegotiationFailed()

    _param2verb = {
        'createstoredquery': gws.OwsVerb.CreateStoredQuery,
        'describecoverage': gws.OwsVerb.DescribeCoverage,
        'describefeaturetype': gws.OwsVerb.DescribeFeatureType,
        'describelayer': gws.OwsVerb.DescribeLayer,
        'describerecord': gws.OwsVerb.DescribeRecord,
        'describestoredqueries': gws.OwsVerb.DescribeStoredQueries,
        'dropstoredquery': gws.OwsVerb.DropStoredQuery,
        'getcapabilities': gws.OwsVerb.GetCapabilities,
        'getfeature': gws.OwsVerb.GetFeature,
        'getfeatureinfo': gws.OwsVerb.GetFeatureInfo,
        'getfeaturewithlock': gws.OwsVerb.GetFeatureWithLock,
        'getlegendgraphic': gws.OwsVerb.GetLegendGraphic,
        'getmap': gws.OwsVerb.GetMap,
        'getprint': gws.OwsVerb.GetPrint,
        'getpropertyvalue': gws.OwsVerb.GetPropertyValue,
        'getrecordbyid': gws.OwsVerb.GetRecordById,
        'getrecords': gws.OwsVerb.GetRecords,
        'gettile': gws.OwsVerb.GetTile,
        'liststoredqueries': gws.OwsVerb.ListStoredQueries,
        'lockfeature': gws.OwsVerb.LockFeature,
        'transaction': gws.OwsVerb.Transaction,
    }

    def requested_operation(self, param_names: str) -> gws.OwsOperation:
        """Return the requested operation.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The operation.

        Raises:
            ``error.OperationNotSupported``: If the operation is not supported by the service.
        """

        _, val = self._get_param(param_names, '')
        op = self.find_operation(val)
        if op:
            return op
        raise error.OperationNotSupported(val)

    def find_operation(self, param: str) -> Optional[gws.OwsOperation]:
        """Find a supported operation by its name.

        Args:
            param: Operation name, case-insensitive.

        Returns:
            The operation, or ``None`` if the service does not support it.
        """

        verb = self._param2verb.get(param.lower())
        if not verb:
            return

        for op in self.service.supportedOperations:
            if op.verb == verb:
                return op

    def requested_crs(self, param_names: str) -> Optional[gws.Crs]:
        """Return the requested CRS.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The CRS, or ``None`` if not requested.

        Raises:
            ``error.InvalidCRS``: If the CRS is unknown or not supported by the service.
        """

        _, val = self._get_param(param_names, '')
        if not val:
            return

        crs = gws.lib.crs.get(val)
        if not crs:
            raise error.InvalidCRS()

        for b in self.service.supportedBounds:
            if crs == b.crs:
                return crs

        raise error.InvalidCRS()

    def requested_bounds(self, param_names: str) -> Optional[gws.Bounds]:
        """Return the requested bounding box, transformed to ``crs``.

        Uses ``crs`` as the default CRS and ``alwaysXY`` for the axis order.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The bounds, or ``None`` if not requested.

        Raises:
            ``error.InvalidParameterValue``: If the bounding box is invalid.
        """

        # OGC 06-042, 7.2.3.5
        # OGC 00-028, 6.2.8.2.3

        p, val = self._get_param(param_names, '')
        if not val:
            return

        bounds = gws.lib.bounds.from_request_bbox(val, default_crs=self.crs, always_xy=self.alwaysXY)
        if bounds:
            return gws.lib.bounds.transform(bounds, self.crs)

        raise error.InvalidParameterValue(p)

    def requested_format(self, param_names: str) -> str:
        """Return the requested format with whitespace removed.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The format, or an empty string if not requested.
        """

        _, val = self._get_param(param_names, '')
        if val:
            # NB our mime types do not contain spaces
            return ''.join(val.split())
        return ''

    def requested_feature_count(self, param_names: str) -> int:
        """Return the requested feature count, limited by the service maximum.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The feature count, or the service default if not requested or not positive.

        Raises:
            ``error.InvalidParameterValue``: If the value is not an integer.
        """

        s = self.int_param(param_names, default=0)
        if s <= 0:
            return self.service.defaultFeatureCount
        return min(self.service.maxFeatureCount, s)

    def requested_xmlns_replacements(self):
        """Read custom namespace prefixes from the ``NAMESPACES`` parameter.

        Returns:
            A dict mapping namespace uris to prefixes.
        """

        s = self.string_param('NAMESPACES', default='')
        if not s:
            return {}

        # OGC 09-025r1, Table 7
        # xmlns(xml,http://www.w3.org/XML/1998/namespace),xmlns(ns37,https://our-ns),xmlns(wfs ... etc

        d = {}

        for xmlns, uri in re.findall(r'xmlns\((.+?),(.+?)\)', s):
            d[uri.strip()] = xmlns.strip()

        return d

    ##

    def _get_param(self, param_names, default):
        """Return the first present parameter as ``(name, value)``; raise ``MissingParameterValue`` if absent and ``default`` is ``None``."""
        names = gws.u.to_list(param_names.upper())

        for p in names:
            if p not in self.params:
                continue
            val = self.params[p]
            return p, val

        if default is not None:
            return '', default

        raise error.MissingParameterValue(names[0])

    def string_param(self, param_names: str, values: Optional[set[str]] = None, default: Optional[str] = None) -> str:
        """Return a string parameter.

        Args:
            param_names: Comma-separated parameter names to look for.
            values: Allowed values, in lower case. If given, the value is lower-cased and checked.
            default: Default value. If ``None``, the parameter is required.

        Returns:
            The parameter value.

        Raises:
            ``error.MissingParameterValue``: If the parameter is required and missing.
            ``error.InvalidParameterValue``: If the value is not allowed.
        """

        p, val = self._get_param(param_names, default)
        if values:
            val = val.lower()
            if val not in values:
                raise error.InvalidParameterValue(p)
        return val

    def list_param(self, param_names: str) -> list[str]:
        """Return a comma-separated parameter as a list.

        Args:
            param_names: Comma-separated parameter names to look for.

        Returns:
            The list of values, empty if the parameter is missing.
        """

        _, val = self._get_param(param_names, '')
        return gws.u.to_list(val)

    def int_param(self, param_names: str, default: Optional[int] = None) -> int:
        """Return an integer parameter.

        Args:
            param_names: Comma-separated parameter names to look for.
            default: Default value. If ``None``, the parameter is required.

        Returns:
            The parameter value.

        Raises:
            ``error.MissingParameterValue``: If the parameter is required and missing.
            ``error.InvalidParameterValue``: If the value is not an integer.
        """

        p, val = self._get_param(param_names, default)
        try:
            return int(val)
        except ValueError:
            raise error.InvalidParameterValue(p)

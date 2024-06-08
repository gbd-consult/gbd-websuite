"""Service Request object."""

from typing import Optional, Callable

import gws
import gws.base.layer.core
import gws.base.legend
import gws.base.model
import gws.base.web
import gws.gis.extent
import gws.gis.render
import gws.lib.mime
import gws.gis.bounds
import gws.gis.crs
import gws.lib.image
import gws.lib.uom

from . import core, layer_caps, error


class TemplateArgs(gws.TemplateArgs):
    """Arguments for service templates."""

    featureCollection: core.FeatureCollection
    layerCapsList: list[core.LayerCaps]
    sr: 'Object'
    service: gws.OwsService
    serviceUrl: str
    url_for: Callable
    gmlVersion: int
    version: str
    intVersion: int
    tileMatrixSets: list[gws.TileMatrixSet]


class Object:
    alwaysXY: bool
    bounds: gws.Bounds
    crs: gws.Crs
    isSoap: bool = False
    layerCapsList: list[core.LayerCaps]
    operation: gws.OwsOperation
    project: gws.Project
    req: gws.WebRequester
    service: gws.OwsService
    targetCrs: gws.Crs
    version: str
    xmlElement: Optional[gws.XmlElement]

    def __init__(self, service: gws.OwsService, req: gws.WebRequester):
        self.service = service
        self.req = req
        self.project = self.requested_project()

        self.operation = self.requested_operation('REQUEST')
        self.version = self.requested_version('VERSION,ACCEPTVERSIONS')

        cache_key = 'layer_caps_' + gws.u.sha256([self.service.uid, self.project.uid, sorted(self.req.user.roles)])
        self.layerCapsList = gws.u.get_app_global(cache_key, self.enum_layer_caps)

        self.alwaysXY = False
        self.isSoap = False
        self.pxWidth = 0
        self.pxHeight = 0
        self.xResolution = 0
        self.yResolution = 0

        # OGC 06-042, 7.2.3.5
        if self.service.updateSequence:
            s = self.string_param('UPDATESEQUENCE', default='')
            if s and s == self.service.updateSequence:
                raise error.CurrentUpdateSequence()
            if s and s > self.service.updateSequence:
                raise error.InvalidUpdateSequence()

    def enum_layer_caps(self):
        lcs = []
        root_layer = self.service.rootLayer or self.project.map.rootLayer
        self._enum_layer_caps(root_layer, lcs, [])
        return lcs

    def _enum_layer_caps(self, layer: gws.Layer, lcs: list[core.LayerCaps], stack: list[core.LayerCaps]):
        if not self.req.user.can_read(layer) or not layer.isEnabledForOws:
            return

        is_suitable = self.service.layer_is_suitable(layer)
        if not is_suitable and not layer.isGroup:
            return

        lc = layer_caps.for_layer(layer, self.req.user, self.service)

        # NB groups must be inspected even if not 'suitable'
        if layer.isGroup:
            lc.isGroup = True
            n = len(lcs)
            for sub_layer in layer.layers:
                self._enum_layer_caps(sub_layer, lcs, stack + [lc])
            if not lc.children:
                # no empty groups
                return
            if is_suitable:
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

    def requested_project(self) -> gws.Project:
        # services can be configured globally (in which case, service.project == None)
        # and applied to multiple projects with the projectUid param
        # or, configured just for a single project (service.project != None)

        p = self.req.param('projectUid')
        if p:
            project = self.req.user.require_project(p)
            if self.service.project and project != self.service.project:
                raise gws.NotFoundError(f'ows {self.service.uid=}: wrong project={p!r}')
            return project

        if self.service.project:
            # for in-project services, ensure the user can access the project
            return self.req.user.require_project(self.service.project.uid)

        raise gws.NotFoundError(f'project not found for {self.service}')

    def requested_version(self, param_names: str) -> str:
        p, val = self._get_param(param_names, default='')
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
        p, val = self._get_param(param_names, default=None)
        verb = self._param2verb.get(val.lower())
        if not verb:
            raise error.InvalidParameterValue(p)

        for op in self.service.supportedOperations:
            if op.verb == verb:
                return op

        raise error.OperationNotSupported()

    def requested_crs(self, param_names: str) -> Optional[gws.Crs]:
        _, val = self._get_param(param_names, default='')
        if not val:
            return

        crs = gws.gis.crs.get(val)
        if not crs:
            raise error.InvalidCRS()

        for b in self.service.supportedBounds:
            if crs == b.crs:
                return crs

        raise error.InvalidCRS()

    def requested_bounds(self, param_names: str) -> Optional[gws.Bounds]:
        # OGC 06-042, 7.2.3.5
        # OGC 00-028, 6.2.8.2.3

        p, val = self._get_param(param_names, '')
        if not val:
            return

        bounds = gws.gis.bounds.from_request_bbox(val, default_crs=self.crs, always_xy=self.alwaysXY)
        if bounds:
            return gws.gis.bounds.transform(bounds, self.crs)

        raise error.InvalidParameterValue(p)

    def requested_format(self, param_names: str) -> str:
        s = self.string_param(param_names, default='').strip()
        if s:
            # NB our mime types do not contain spaces
            return ''.join(s.split())
        return ''

    def requested_feature_count(self, param_names: str) -> int:
        s = self.int_param(param_names, default=0)
        if s <= 0:
            return self.service.defaultFeatureCount
        return min(self.service.maxFeatureCount, s)

    ##

    def _get_param(self, param_names, default):
        names = gws.u.to_list(param_names.upper())

        for p in names:
            if not self.req.has_param(p):
                continue
            val = self.req.param(p)
            return p, val

        if default is not None:
            return '', default

        raise error.MissingParameterValue(names[0])

    def string_param(self, param_names: str, values: Optional[set[str]] = None, default: Optional[str] = None) -> str:
        p, val = self._get_param(param_names, default)
        if values:
            val = val.lower()
            if val not in values:
                raise error.InvalidParameterValue(p)
        return val

    def list_param(self, param_names: str) -> list[str]:
        _, val = self._get_param(param_names, '')
        return gws.u.to_list(val)

    def int_param(self, param_names: str, default: Optional[int] = None) -> int:
        p, val = self._get_param(param_names, default)
        try:
            return int(val)
        except ValueError:
            raise error.InvalidParameterValue(p)

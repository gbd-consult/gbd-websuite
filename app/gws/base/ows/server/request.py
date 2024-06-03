"""Service Request object."""

from typing import Optional, Callable, cast

import gws
import gws.base.layer.core
import gws.base.legend
import gws.base.model
import gws.base.web
import gws.gis.extent
import gws.gis.render
import gws.lib.date
import gws.lib.mime
import gws.gis.bounds
import gws.gis.crs
import gws.lib.image
import gws.lib.uom

from . import core, layer_caps, error

OWS_VERBS = [
    gws.OwsVerb.CreateStoredQuery,
    gws.OwsVerb.DescribeCoverage,
    gws.OwsVerb.DescribeFeatureType,
    gws.OwsVerb.DescribeLayer,
    gws.OwsVerb.DescribeRecord,
    gws.OwsVerb.DescribeStoredQueries,
    gws.OwsVerb.DropStoredQuery,
    gws.OwsVerb.GetCapabilities,
    gws.OwsVerb.GetFeature,
    gws.OwsVerb.GetFeatureInfo,
    gws.OwsVerb.GetFeatureWithLock,
    gws.OwsVerb.GetLegendGraphic,
    gws.OwsVerb.GetMap,
    gws.OwsVerb.GetPrint,
    gws.OwsVerb.GetPropertyValue,
    gws.OwsVerb.GetRecordById,
    gws.OwsVerb.GetRecords,
    gws.OwsVerb.GetTile,
    gws.OwsVerb.ListStoredQueries,
    gws.OwsVerb.LockFeature,
    gws.OwsVerb.Transaction,
]


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
    project: gws.Project
    req: gws.WebRequester
    service: gws.OwsService
    targetCrs: gws.Crs
    verb: gws.OwsVerb
    version: str
    xmlElement: Optional[gws.XmlElement]

    def __init__(self, service: gws.OwsService, req: gws.WebRequester):
        self.req = req
        self.service = service
        self.alwaysXY = False
        self.project = self.get_project()
        self.isSoap = False
        self.verb = self.get_verb()
        self.version = self.get_version()
        self.layerCapsList = self.get_layer_caps()

        self.pxWidth = 0
        self.pxHeight = 0
        self.xResolution = 0
        self.yResolution = 0

        # OGC 06-042, 7.2.3.5
        if self.service.updateSequence:
            s = self.string_param('UPDATESEQUENCE', default='')
            if s == self.service.updateSequence:
                raise error.CurrentUpdateSequence()
            if s > self.service.updateSequence:
                raise error.InvalidUpdateSequence()

    def get_layer_caps(self) -> list[core.LayerCaps]:
        key = gws.u.sha256([self.service.uid, self.project.uid, sorted(self.req.user.roles)])
        return gws.u.get_app_global(key, self.enum_layer_caps)

    def enum_layer_caps(self):
        lcs = []
        self.collect_layer_caps(
            self.service.rootLayer or self.project.map.rootLayer,
            lcs,
            []
        )
        return lcs

    def collect_layer_caps(self, layer: gws.Layer, lcs: list[core.LayerCaps], stack: list[core.LayerCaps]):
        if not self.req.user.can_read(layer) or not layer.isEnabledForOws:
            return

        can_use = self.service.layer_is_suitable(layer)
        if not can_use and not layer.isGroup:
            return

        lc = layer_caps.for_layer(layer, self.req.user, self.service)

        if layer.isGroup:
            lc.isGroup = True
            n = len(lcs)
            for sub_layer in layer.layers:
                self.collect_layer_caps(sub_layer, lcs, stack + [lc])
            if not lc.children:
                return
            if can_use:
                lc.hasLegend = any(c.hasLegend for c in lc.children)
                lc.hasSearch = any(c.hasSearch for c in lc.children)
                lcs.insert(n, lc)
        else:
            lc.isGroup = False
            lcs.append(lc)
            for sup_lc in stack:
                sup_lc.leaves.append(lc)

        if stack:
            stack[-1].children.append(lc)

    def get_project(self) -> gws.Project:
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

    def get_version(self) -> str:
        s = self.string_param('VERSION,ACCEPTVERSIONS', default='')
        if not s:
            # the first supported version is the default
            return self.service.supportedVersions[0]

        for v in gws.u.to_list(s):
            for ver in self.service.supportedVersions:
                if ver.startswith(v):
                    return ver

        raise error.VersionNegotiationFailed()

    def get_verb(self) -> gws.OwsVerb:
        s = self.string_param('REQUEST', default='')
        if not s:
            raise error.MissingParameterValue('REQUEST')

        for verb in OWS_VERBS:
            if verb.lower() == s.lower():
                return verb
        raise error.InvalidParameterValue('REQUEST')

    def get_crs(self, name: str) -> Optional[gws.Crs]:
        s = self.string_param(name, default='')
        if not s:
            return

        crs = gws.gis.crs.get(s)
        if not crs:
            raise error.InvalidCRS()

        for b in self.service.supportedBounds:
            if crs == b.crs:
                return crs

        raise error.InvalidCRS()

    def get_bounds(self, name: str) -> Optional[gws.Bounds]:
        # OGC 06-042, 7.2.3.5
        # OGC 00-028, 6.2.8.2.3

        p, val = self._get_param(name, '')
        if not val:
            return

        bounds = gws.gis.bounds.from_request_bbox(val, default_crs=self.crs, always_xy=self.alwaysXY)
        if bounds:
            return gws.gis.bounds.transform(bounds, self.crs)

        raise error.InvalidParameterValue(p)

    def get_feature_count(self, name: str) -> int:
        s = self.int_param(name, default=0)
        if s <= 0:
            return self.service.defaultFeatureCount
        return min(self.service.maxFeatureCount, s)

    def _get_param(self, name, default):
        names = gws.u.to_list(name.upper())

        for n in names:
            if not self.req.has_param(n):
                continue
            val = self.req.param(n)
            return n, val

        if default is not None:
            return '', default

        raise error.MissingParameterValue(names[0])

    def string_param(self, name: str, values: Optional[set[str]] = None, default: Optional[str] = None) -> str:
        p, val = self._get_param(name, default)
        if values:
            val = val.lower()
            if val not in values:
                raise error.InvalidParameterValue(p)
        return val

    def list_param(self, name: str) -> list[str]:
        _, val = self._get_param(name, '')
        return gws.u.to_list(val)

    def int_param(self, name: str, default: Optional[int] = None) -> int:
        p, val = self._get_param(name, default)
        try:
            return int(val)
        except ValueError:
            raise error.InvalidParameterValue(p)

    ##

    def feature_collection(self, lcs: list[core.LayerCaps], hits: int, results: list[gws.SearchResult]) -> core.FeatureCollection:
        fc = core.FeatureCollection(
            members=[],
            timestamp=gws.lib.date.now_iso(with_tz=False),
            numMatched=hits,
            numReturned=len(results),
        )

        lcs_map = {id(lc.layer): lc for lc in lcs}

        for r in results:
            r.feature.transform_to(self.targetCrs)
            fc.members.append(core.FeatureCollectionMember(
                feature=r.feature,
                layer=r.layer,
                layerCaps=lcs_map.get(id(r.layer)) if r.layer else None
            ))

        return fc

    def render_legend(self, lcs: list[core.LayerCaps]) -> gws.ContentResponse:
        legend = cast(gws.Legend, self.service.root.create_temporary(
            gws.ext.object.legend,
            type='combined',
            layerUids=[lc.layer.uid for lc in lcs]))

        content = None

        lro = legend.render()
        if lro:
            content = gws.base.legend.output_to_bytes(lro)

        return gws.ContentResponse(
            mime=gws.lib.mime.PNG,
            content=content or gws.lib.image.empty_pixel()
        )

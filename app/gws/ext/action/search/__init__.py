import gws
import gws.web
import gws.gis.shape
import gws.gis.feature
import gws.tools.json2
import gws.common.template

import gws.types as t

MAX_LIMIT = 1000


class Config(t.WithTypeAndAccess):
    """search action"""
    limit: int = 1000  #: search results limit
    pixelTolerance: int = 5  #: pixel tolerance for geometry searches


class Response(t.Response):
    features: t.List[t.FeatureProps]
    total: int


class Params(t.Data):
    bbox: t.Extent
    keyword: str = ''
    layerUids: t.List[str]
    pixelTolerance: int = 10
    limit: int = 0
    projectUid: str
    resolution: float
    shape: t.Optional[t.ShapeProps]
    withAttributes: bool = True
    withDescription: bool = True
    withGeometry: bool = True


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.limit = self.var('limit')
        self.pixel_tolerance = self.var('pixelTolerance')
        self.feature_format = self.create_shared_object(
            'gws.common.format',
            'default_feature_description',
            gws.common.template.builtin_config('feature_format')
        )

    def api_find_features(self, req, p: Params) -> Response:
        """Perform a search"""

        project = req.require_project(p.projectUid)
        layers = gws.compact(req.acquire('gws.ext.gis.layer', uid) for uid in p.layerUids)
        limit = min(p.limit, self.limit) if p.limit else self.limit

        args = t.SearchArgs({
            'bbox': p.bbox,
            'crs': project.map.crs,
            'keyword': (p.keyword or '').strip(),
            'layers': layers,
            'project': project,
            'resolution': p.resolution,
            'tolerance': self.pixel_tolerance * p.resolution,
            'shape': gws.gis.shape.from_props(p.shape) if p.get('shape') else None,
        })

        # layer-provider-feature triples
        lpf = []

        for prov in project.get_children('gws.ext.search.provider'):
            args.limit = limit - len(lpf)
            if args.limit <= 0:
                break
            for f in self.do_search(req, None, prov, args):
                lpf.append((None, prov, f))

        for layer in layers:
            args.limit = limit - len(lpf)
            if args.limit <= 0:
                break
            for prov in layer.get_children('gws.ext.search.provider'):
                for f in self.do_search(req, layer, prov, args):
                    lpf.append((layer, prov, f))

        total = len(lpf)
        fprops = []

        for layer, prov, f in lpf[:limit]:
            f.provider = prov
            f.layer = layer

            f.transform(project.map.crs)

            s = prov.title
            if not s and layer:
                s = layer.title
            if s:
                f.category = s

            fmt = prov.feature_format
            if not fmt and layer:
                fmt = layer.feature_format
            if not fmt:
                fmt = self.feature_format

            f.apply_format(fmt)

            p = f.props
            delattr(p, 'attributes')

            fprops.append(p)

        return Response({
            'features': fprops,
            'total': total,
        })

    def do_search(self, req, layer, prov: t.SearchProviderInterface, args: t.SearchArgs):
        gws.log.debug(
            'SEARCH_BEGIN: prov=%r layer=%r limit=%d' % (gws.get(prov, 'uid'), gws.get(layer, 'uid'), args.limit))

        if not req.user.can('execute', prov):
            gws.log.debug('SEARCH_END: NO_ACCESS')
            return []

        if not prov.can_run(args):
            gws.log.debug(f'SEARCH_END: N_A')
            return []

        res = prov.run(layer, args)
        gws.log.debug('SEARCH_END')
        return res

import gws
import gws.lib.crs
import gws.lib.gis.source
import gws.lib.metadata
import gws.lib.mime
import gws.lib.ows
import gws.types as t


class OperationConfig(gws.Config):
    formats: t.Optional[t.List[str]]
    url: gws.Url
    postUrl: t.Optional[gws.Url]
    verb: gws.OwsVerb


class ProviderConfig(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    invertAxis: t.Optional[t.List[gws.CrsId]]  #: projections that are known to have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    operations: t.Optional[t.List[OperationConfig]]
    forceCrs: t.Optional[gws.CrsId]  #: use this CRS for requests
    sourceLayers: t.Optional[gws.lib.gis.source.LayerFilterConfig]  #: source layers to use
    url: gws.Url  #: service url


class Caps(gws.Data):
    metadata: gws.lib.metadata.Metadata
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.SourceLayer]
    version: str


class Provider(gws.Node, gws.IOwsProvider):
    inverted_crs: t.List[gws.ICrs]
    preferred_formats: t.Dict[gws.OwsVerb, t.Optional[str]]

    def configure(self):
        self.force_crs = gws.lib.crs.get(self.var('forceCrs'))
        self.inverted_crs = gws.compact(gws.lib.crs.get(c) for c in self.var('invertAxis', default=[]))
        self.source_layers = []
        self.url = self.var('url')
        self.version = ''

        # operations from config, if any + a mandatory Caps operation.
        # operations from the caps document will be added to this list,
        # so that configured ops take precedence

        self.operations = []

        for cfg in self.var('operations', default=[]):
            self.operations.append(gws.OwsOperation(
                formats=cfg.get('formats', []),
                get_url=cfg.get('url'),
                post_url=cfg.get('postUrl'),
                verb=cfg.get('verb'),
                params={},
            ))

        if all(op.verb != gws.OwsVerb.GetCapabilities for op in self.operations):
            self.operations.insert(0, gws.OwsOperation(
                formats=['text/xml'],
                get_url=self.var('url'),
                post_url=None,
                verb=gws.OwsVerb.GetCapabilities,
                params={},
            ))

        self.preferred_formats = {}

        image_ops = {
            gws.OwsVerb.GetLegendGraphic,
            gws.OwsVerb.GetMap,
            gws.OwsVerb.GetTile,
        }

        def _best_format(op):
            best = gws.lib.mime.PNG if op.verb in image_ops else gws.lib.mime.XML

            # they support exactly what we want...
            for fmt in op.formats:
                if gws.lib.mime.get(fmt) == best:
                    return fmt
            # ...or they support anything at all...
            for fmt in op.formats:
                return fmt
            # ...otherwise, no preferred format
            return None

        for op in self.operations:
            self.preferred_formats[op.verb] = _best_format(op)

    def operation(self, verb: gws.OwsVerb, method='GET'):
        for op in self.operations:
            if op.verb == verb and op.get(method.lower() + '_url'):
                return op

    def operation_args(self, verb: gws.OwsVerb, method='GET', params=None):
        op = self.operation(verb, method)
        if not op:
            raise gws.Error(f'operation not found: {verb!r}')

        return {
            'url': op.get(method.lower() + '_url'),
            'protocol': self.protocol,
            'verb': verb,
            'params': self.operation_params(op, params or {}),
        }

    def operation_params(self, op: gws.OwsOperation, params: dict) -> dict:
        """Merge Operation.params and request params."""

        pmap = {
            name.upper(): [name, allowed, None]
            for name, allowed in op.params.items()
        }

        for name, val in params.items():
            name = name.upper()
            if name in pmap:
                pmap[name][2] = val
            else:
                pmap[name] = [name, None, val]

        res = {}

        for name, allowed, val in pmap.values():
            if not allowed or val in allowed:
                res[name] = val
            elif val is None:
                res[name] = allowed[0]
            else:
                raise gws.Error(f'invalid param {name!r}, value={val!r}, allowed={allowed!r}')

        return res

    def get_capabilities(self):
        return gws.lib.ows.request.get_text(
            **self.operation_args(gws.OwsVerb.GetCapabilities),
            max_age=self.var('capsCacheMaxAge'))

    def find_features(self, args: gws.SearchArgs, source_layers: t.List[gws.SourceLayer]) -> t.List[gws.IFeature]:
        return []

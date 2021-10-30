import gws
import gws.lib.gis
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
    operations: t.Optional[t.List[OperationConfig]]
    invertAxis: t.Optional[t.List[gws.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceCrs: t.Optional[gws.Crs]  #: use this CRS for requests
    url: gws.Url  #: service url


class Provider(gws.Node, gws.IOwsProvider):
    invert_axis_crs: t.List[str]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_crs: gws.Crs
    operations: t.List[gws.OwsOperation]
    preferred_formats: t.Dict[str, t.Optional[str]]

    def configure(self):
        self.invert_axis_crs = self.var('invertAxis', default=[])

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
            self.preferred_formats[str(op.verb)] = _best_format(op)

        self.source_crs = self.var('sourceCrs')
        self.source_layers = []
        self.supported_crs = []
        self.url = self.var('url')
        self.version = ''

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

        ps = {
            name.upper(): [name, allowed, None]
            for name, allowed in op.params.items()
        }

        for name, val in params.items():
            name = name.upper()
            if name in ps:
                ps[name][2] = val
            else:
                ps[name] = [name, None, val]

        d = {}

        for name, allowed, val in ps.values():
            if not allowed or val in allowed:
                d[name] = val
            elif val is None:
                d[name] = allowed[0]
            else:
                raise gws.Error(f'invalid param {name!r}, value={val!r}, allowed={allowed!r}')

        return d

    def get_capabilities(self):
        return gws.lib.ows.request.get_text(
            **self.operation_args(gws.OwsVerb.GetCapabilities),
            max_age=self.var('capsCacheMaxAge'))

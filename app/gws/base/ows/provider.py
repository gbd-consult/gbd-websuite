import gws
import gws.lib.gis
import gws.lib.json2
import gws.lib.ows
import gws.types as t


class OperationConfig(gws.Config):
    formats: t.Optional[t.List[str]]
    url: gws.Url
    postUrl: t.Optional[gws.Url]
    verb: gws.OwsVerb


class Config(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    operations: t.Optional[t.List[OperationConfig]]
    invertAxis: t.Optional[t.List[gws.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceCrs: t.Optional[gws.Crs]  #: use this CRS for requests
    url: gws.Url  #: service url


class Object(gws.Object, gws.IOwsProvider):
    invert_axis_crs: t.List[str]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_crs: gws.Crs
    operations: t.List[gws.OwsOperation]

    def configure(self):
        self.invert_axis_crs = self.var('invertAxis', default=[])

        self.operations = [gws.OwsOperation(
            formats=['text/xml'],
            get_url=self.var('url'),
            post_url=None,
            verb=gws.OwsVerb.GetCapabilities,
        )]

        for cfg in self.var('operations', default=[]):
            self.operations.append(gws.OwsOperation(
                formats=cfg.get('formats', []),
                get_url=cfg.get('url'),
                post_url=cfg.get('postUrl'),
                verb=cfg.get('verb'),
            ))

        self.source_crs = self.var('sourceCrs')
        self.source_layers = []
        self.supported_crs = []
        self.url = self.var('url')
        self.version = ''

        p: gws.OwsVerb = gws.OwsVerb.GetCapabilities

    def operation(self, verb: gws.OwsVerb, method='GET'):
        for op in reversed(self.operations):
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
            'params': gws.merge({}, op.params, params),
        }

    def get_capabilities(self):
        return gws.lib.ows.request.get_text(
            **self.operation_args(gws.OwsVerb.GetCapabilities),
            max_age=self.var('capsCacheMaxAge'))

import gws
import gws.lib.gis
import gws.lib.json2
import gws.lib.ows
import gws.types as t


def shared_object(root, klass, cfg):
    uid = cfg.get('url')
    params = cfg.get('params')
    if params:
        uid += '_' + gws.lib.json2.to_hash(params)
    return root.create_shared_object(klass, uid, cfg)


class Config(gws.Config):
    capsCacheMaxAge: gws.Duration = '1d'  #: max cache age for capabilities documents
    capsParams: t.Optional[t.Dict]  #: additional parameters for GetCapabilities requests
    invertAxis: t.Optional[t.List[gws.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    sourceCrs: t.Optional[gws.Crs]  #: use this CRS for requests
    url: gws.Url  #: service url


class Object(gws.Object, gws.IOwsProvider):
    invert_axis_crs: t.List[str]
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    source_crs: gws.Crs

    def configure(self):
        self.invert_axis_crs = self.var('invertAxis', default=[])
        self.operations = []
        self.service_version = ''
        self.source_crs = self.var('sourceCrs')
        self.source_layers = []
        self.supported_crs = []
        self.url = self.var('url')

    def operation(self, name: str) -> t.Optional[gws.OwsOperation]:
        for op in self.operations:
            if op.name == name:
                return op

    def get_capabilities(self):
        return gws.lib.ows.request.get_text(
            self.url,
            service=self.service_type,
            verb='GetCapabilities',
            params=self.var('capsParams'),
            max_age=self.var('capsCacheMaxAge'))

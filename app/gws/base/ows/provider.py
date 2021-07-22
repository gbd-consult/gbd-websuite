import gws
import gws.lib.gis
import gws.lib.json2
import gws.types as t


def shared_object(root, klass, cfg):
    uid = cfg.get('url')
    params = cfg.get('params')
    if params:
        uid += '_' + gws.lib.json2.to_hash(params)
    return root.create_shared_object(klass, uid, cfg)


class Config(gws.Config):
    invertAxis: t.Optional[t.List[gws.Crs]]  #: projections that have an inverted axis (yx)
    maxRequests: int = 0  #: max concurrent requests to this source
    url: gws.Url  #: service url


class Object(gws.Node, gws.IOwsProvider):
    metadata: gws.IMetaData
    operations: t.List[gws.OwsOperation]
    service_type: str
    service_version: str
    supported_crs: t.List[gws.Crs]
    url: gws.Url

    invert_axis_crs: t.List[str]
    source_layers: t.List[gws.lib.gis.SourceLayer]

    def configure(self):
        self.invert_axis_crs = self.var('invertAxis', default=[])
        self.operations = []
        self.source_layers = []
        self.supported_crs = []
        self.url = self.var('url')
        self.service_version = ''

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        pass

    def operation(self, name: str) -> t.Optional[gws.OwsOperation]:
        for op in self.operations:
            if op.name == name:
                return op

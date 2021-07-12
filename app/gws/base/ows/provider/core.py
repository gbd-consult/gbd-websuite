import gws
import gws.types as t
import gws.lib.source



class OwsOperation(gws.Data):
    formats: t.List[str]
    get_url: gws.Url
    name: str
    parameters: dict
    post_url: gws.Url


class Object(gws.Node, gws.IOwsService):
    invert_axis_crs: t.List[str]
    operations: t.List[OwsOperation]
    source_layers: t.List[gws.lib.source.Layer]
    service_type: str
    version: str
    meta: gws.IMeta
    supported_crs: t.List[gws.Crs]
    url: gws.Url

    def configure(self):
        self.invert_axis_crs = self.var('invertAxis', default=[])
        self.operations = []
        self.source_layers: t.List[gws.lib.source.Layer] = []
        self.supported_crs = []
        self.url = self.var('url')

    def find_features(self, args: gws.SearchArgs) -> t.List[gws.IFeature]:
        pass

    def operation(self, name: str) -> t.Optional[OwsOperation]:
        for op in self.operations:
            if op.name == name:
                return op

import gws
import gws.types as t


#:export IOwsProvider
class Object(gws.Object, t.IOwsProvider):
    def __init__(self):
        super().__init__()

        self.invert_axis_crs: t.List[str] = []
        self.meta: t.MetaData = None
        self.operations: t.List[t.OwsOperation] = []
        self.source_layers: t.List[t.SourceLayer] = []
        self.supported_crs: t.List[t.Crs] = []
        self.type: str = ''
        self.url: t.Url = ''
        self.version: str = ''

    def configure(self):
        super().configure()

        self.invert_axis_crs = self.var('invertAxis')
        self.url = self.var('url')


    def find_features(self, args: t.SearchArgs) -> t.List[t.IFeature]:
        pass

    def operation(self, name: str) -> t.OwsOperation:
        for op in self.operations:
            if op.name == name:
                return op

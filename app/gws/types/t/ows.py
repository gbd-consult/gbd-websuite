### OWS providers and services.

from .base import List, Dict, Extent, Crs, Url
from .object import Object
from .meta import MetaData
from .search import SearchArguments
from .feature import Feature


class SourceStyle:
    def __init__(self):
        self.is_default = False
        self.legend: Url = ''
        self.meta = MetaData()


class SourceLayer:
    def __init__(self):
        self.data_source = {}

        self.supported_crs: List[Crs] = []
        self.extents: Dict[Crs, Extent] = {}

        self.is_expanded = False
        self.is_group = False
        self.is_image = False
        self.is_queryable = False
        self.is_visible = False

        self.layers: List['SourceLayer'] = []

        self.meta = MetaData()
        self.name = ''
        self.title = ''

        self.opacity = 1
        self.scale_range: List[float] = []
        self.styles: List[SourceStyle] = []
        self.legend = ''
        self.resource_urls = {}

        self.a_path = ''
        self.a_uid = ''
        self.a_level = 0


class OwsOperation:
    def __init__(self):
        self.name = ''
        self.formats: List[str] = []
        self.get_url: Url = ''
        self.post_url: Url = ''
        self.parameters: dict = {}


class OwsProviderObject(Object):
    meta: 'MetaData'
    operations: List[OwsOperation]
    source_layers: List[SourceLayer]
    supported_crs: List[Crs]
    type: str
    url: Url
    version: str

    def find_features(self, args: 'SearchArguments') -> List[Feature]:
        pass

    def operation(self, name: str) -> OwsOperation:
        pass


class OwsServiceObject(Object):
    def __init__(self):
        super().__init__()
        self.name: str = ''
        self.meta: 'MetaData' = None
        self.type: str = ''
        self.version: str = ''

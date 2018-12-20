import typing

import shapely.geometry

from .data import Data

Optional = typing.Optional
List = typing.List
Dict = typing.Dict
Tuple = typing.Tuple
Union = typing.Union
cast = typing.cast

#: alias
Extent = Tuple[float, float, float, float]

#: alias
Point = Tuple[float, float]

#: alias
Size = Tuple[float, float]

Config = Data


# NB: we cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum:
    pass


class literal:
    pass


# dummy classes to support extension typing
class ext:
    class action:
        class Config(Config):
            pass

    class auth:
        class provider:
            class Config(Config):
                pass

    class template:
        class Config(Config):
            pass

        class Props(Data):
            pass

    class db:
        class provider:
            class Config(Config):
                pass

    class gis:
        class layer:
            class Config(Config):
                pass

            class LayerProps(Data):
                pass

        class source:
            class Config(Config):
                pass

    class search:
        class provider:
            class Config(Config):
                pass


class filepath:
    """valid readable file path on the server"""
    pass


class dirpath:
    """valid readable directory path on the server"""
    pass


class duration:
    """string like "1w 2d 3h 4m 5s" or an int number of seconds"""
    pass


class regex:
    """regular expression, as used in Python"""
    pass


class formatstr:
    """string with {attribute} placeholders"""
    pass


class crsref:
    """CRS code like "EPSG:3857" """
    pass


class url:
    """http or https URL"""
    pass


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessMode(Enum):
    read = 'read'
    write = 'write'
    execute = 'execute'


class AccessRuleConfig(Config):
    """access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    mode: List[AccessMode] = ['read', 'write', 'execute']  #: access mode
    role: List[str]  #: list of roles to which this rule applies


#: alias
Access = List[AccessRuleConfig]


class ResponseError(Data):
    status: int
    info: str


class Response(Data):
    error: Optional[ResponseError]


class NoParams(Data):
    pass


class HttpResponse(Data):
    mimeType: str
    content: str


class WithType(Config):
    type: str  #: object type


class WithTypeAndAccess(WithType):
    access: Optional[Access]  #: access rights


class DocumentRootConfig(Config):
    """assets base directory configuration"""

    dir: dirpath  #: directory path
    allowMime: Optional[List[str]]  #: allowed mime types
    denyMime: Optional[List[str]]  #: disallowed mime types (from the standard list)


class CacheConfig(Config):
    """map cache configuration"""

    enabled: bool = False  #: cache is enabled
    maxAge: duration = '1d'  #: cache max. age
    maxLevel: int = 1  #: max. zoom level to cache
    options: dict = {}  #: additional MapProxy cache options


class GridConfig(Config):
    """grid configuration for tiled or cached map data"""

    origin: str = 'nw'  #: position of the first tile (nw or sw)
    tileSize: int = 256  #: tile size
    metaSize: int = 4  #: number of meta-tiles to fetch
    metaBuffer: int = 200  #: pixel buffer
    options: dict = {}  #: additional MapProxy grid options


class ShapeInterface:
    crs: crsref
    geo: shapely.geometry.base.BaseGeometry
    props: dict
    type: str
    wkb: str
    wkb_hex: str
    wkt: str
    bounds: Extent

    def transform(self, to_crs):
        raise NotImplementedError


class MetaContact(Data):
    area: str = ''
    city: str = ''
    country: str = ''
    email: str = ''
    fax: str = ''
    organization: str = ''
    person: str = ''
    phone: str = ''
    position: str = ''
    zip: str = ''


class MetaData(Data):
    abstract: str = ''
    access_constraints: str = ''
    attribution: str = ''
    contact: MetaContact = {}
    fees: str = ''
    image: url = ''
    keywords: List[str] = []
    name: str = ''
    title: str = ''
    url: url = ''


class MetaConfig(Config):
    """object metadata configuration"""

    abstract: str = ''  #: object abstract description
    attribution: str = ''  #: attribution (copyright) string
    image: url = ''  #: image (logo) url
    images: dict = {}  #: further images
    keywords: List[str] = []  #: keywords
    name: str = ''  #: object internal name
    title: str = ''  #: object title
    url: url = ''  #: object metadata url


class ServiceMetaData(MetaData):
    contact: MetaContact
    access_constraints: str = ''
    fees: str = ''


class ServiceOperation:
    def __init__(self):
        self.name = ''
        self.formats: List[str] = []
        self.get_url: url = ''
        self.post_url: url = ''
        self.parameters: dict = {}


class ServiceQuery:
    crs: str
    max_age: str
    params: Dict
    request: str
    service: 'ServiceInterface'
    url: str
    version: str

    def run(self) -> str:
        raise NotImplementedError


class FindFeaturesArgs(Data):
    axis: str
    bbox: Extent
    count: int
    crs: crsref
    layers: List[str]
    params: Dict
    point: Point
    resolution: float


class ServiceInterface:
    type: str
    layers: List['SourceLayer']
    meta: MetaData
    # @TODO should be Dict[str, ServiceOperation]
    operations: dict
    supported_crs: List[crsref]
    url: str
    version: str

    def find_features(self, args: FindFeaturesArgs) -> List['FeatureInterface']:
        raise NotImplementedError


class Service:
    type: str
    layers: List['SourceLayer']
    meta: MetaData
    # @TODO should be Dict[str, ServiceOperation]
    operations: dict
    supported_crs: List[crsref]
    url: str
    version: str

    def find_features(self, args: FindFeaturesArgs) -> List['FeatureInterface']:
        raise NotImplementedError


class SourceStyle:
    def __init__(self):
        self.is_default = False
        self.legend: url = ''
        self.meta = MetaData()


class SourceLayer:
    def __init__(self):
        self.data_source = {}

        self.supported_crs: List[crsref] = []
        self.extents: Dict[crsref, Extent] = {}

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

        self.a_path = ''
        self.a_uid = ''
        self.a_level = 0


#####

class AttributeConfig(Config):
    """attribute configuration"""

    title: str = ''  #: title
    name: str = ''  #: internal name
    type: str = ''  #: type


class Attribute(Data):
    title: str = ''
    name: str = ''
    type: str = ''


class ObjectInterface:
    children: List['ObjectInterface']
    config: object
    klass: str
    parent: 'ObjectInterface'
    root: 'ObjectInterface'
    uid: str
    props: dict

    def is_a(self, klass):
        raise NotImplementedError

    def initialize(self, cfg):
        raise NotImplementedError

    def configure(self):
        raise NotImplementedError

    def var(self, key, default=None, parent=False):
        raise NotImplementedError

    def add_child(self, klass, cfg):
        raise NotImplementedError

    def get_children(self, klass) -> List['ObjectInterface']:
        raise NotImplementedError

    def get_closest(self, klass) -> 'ObjectInterface':
        raise NotImplementedError

    def find_all(self, klass) -> 'ObjectInterface':
        raise NotImplementedError

    def find_first(self, klass) -> 'ObjectInterface':
        raise NotImplementedError

    def find(self, klass, uid) -> List['ObjectInterface']:
        raise NotImplementedError


class TemplateQualityLevel(Data):
    """named quality level for templates"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class TemplateConfig(Config):
    type: str  #: template type
    qualityLevels: Optional[List[TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: Optional[List[AttributeConfig]]  #: user-editable template attributes
    path: Optional[filepath]  #: path to a template file
    text: str = ''  #: template content
    title: str = ''  #: template title


class TemplateProps(Data):
    uid: str
    title: str
    qualityLevels: List[TemplateQualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: List[Attribute]


class TemplateRenderOutput(Data):
    mimeType: str
    content: str
    path: str


class MapRenderInputItem(Data):
    bitmap: str
    features: List['FeatureInterface']
    layer: 'LayerObject'
    sub_layers: List[str]
    opacity: float
    print_as_vector: bool
    style: 'StyleProps'


class MapRenderInput(Data):
    out_path: str
    bbox: Extent
    rotation: int
    scale: int
    dpi: int
    map_size_px: Size
    items: List[MapRenderInputItem]


class MapRenderOutputItem(Data):
    type: str
    image_path: str = ''
    svg_elements: List[str] = []


class MapRenderOutput(Data):
    bbox: Extent
    dpi: int
    rotation: int
    scale: int
    items: List[MapRenderOutputItem]


# noinspection PyAbstractClass
class TemplateObject(ObjectInterface):
    data_model: List[Attribute]
    map_size: List[int]
    page_size: List[int]

    def dpi_for_quality(self, quality: int) -> int:
        raise NotImplementedError

    def render(self, context: dict, render_output: MapRenderOutput = None,
               out_path: str = None, format: str = None) -> TemplateRenderOutput:
        raise NotImplementedError

    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str:
        raise NotImplementedError


# noinspection PyAbstractClass
class SourceObject(ObjectInterface):
    crs: crsref
    extent: Extent
    layers: List[SourceLayer]
    service: ServiceInterface

    def mapproxy_config(self, mc, options=None):
        raise NotImplementedError

    def service_metadata(self) -> ServiceMetaData:
        raise NotImplementedError

    def layer_metadata(self, layer_name: str) -> MetaData:
        raise NotImplementedError

    def get_features(self, keyword: str, shape: ShapeInterface, sort: str, limit: int):
        raise NotImplementedError

    def modify_features(self, operation, feature_params):
        raise NotImplementedError


class MapView:
    crs: crsref
    extent: Extent
    resolutions: List[float]


class LayerProps(Data):
    pass


# noinspection PyAbstractClass
class LayerObject(ObjectInterface, MapView):
    cache: CacheConfig
    description_template: TemplateObject
    feature_format: 'FormatInterface'
    grid: GridConfig
    legend: url
    map: 'MapObject'
    meta: MetaData
    opacity: float
    parent: MapView
    props: LayerProps
    source: SourceObject
    title: str

    def mapproxy_config(self, mc, options=None):
        raise NotImplementedError

    def render_bbox(self, bbox, width, height, **client_params):
        raise NotImplementedError

    def render_xyz(self, x, y, z):
        raise NotImplementedError

    def render_svg(self, bbox, dpi, scale, rotation, style):
        raise NotImplementedError

    def description(self, options=None) -> str:
        raise NotImplementedError

    def get_features(self, bbox):
        raise NotImplementedError

    def modify_features(self, operation, feature_params):
        raise NotImplementedError

    def search(self, provider: 'SearchProviderInterface', args: 'SearchArgs') -> List['FeatureInterface']:
        raise NotImplementedError


class ShapeProps(Data):
    geometry: dict
    crs: str


class StyleProps(Data):
    """Feature style"""

    type: str  #: style type ("css")
    content: Optional[dict]  #: css rules
    value: Optional[str]  #: raw style content


class FeatureProps(Data):
    attributes: Optional[dict]
    description: str = ''
    label: str = ''
    shape: Optional[ShapeProps]
    style: Optional[StyleProps]
    teaser: str = ''
    title: str = ''
    uid: Optional[str]


class FormatConfig(Config):
    """feature format"""

    description: Optional[TemplateConfig]  #: template for feature descriptions
    category: formatstr = ''  #: feature category
    label: formatstr = ''  #: feature label on the map
    model: Optional[dict]  #: attribute transformation rules
    teaser: Optional[TemplateConfig]  #: template for feature teasers (short descriptions)
    title: formatstr = ''  #: feature title


class FormatInterface(ObjectInterface):
    description: TemplateObject
    category: str
    label: str
    data_model: dict
    teaser: TemplateObject
    title: str


class FeatureInterface:
    attributes: dict
    description: str
    category: str
    label: str
    props: FeatureProps
    shape: ShapeInterface
    shape_props: ShapeProps
    source: ObjectInterface
    style: StyleProps
    teaser: str
    title: str
    uid: str

    def transform(self, to_crs):
        """Transform the feature to another CRS"""
        raise NotImplementedError

    def to_svg(self, bbox, dpi, scale, rotation):
        raise NotImplementedError

    def to_geojs(self):
        raise NotImplementedError

    def apply_format(self, fmt: FormatInterface):
        raise NotImplementedError


# noinspection PyAbstractClass
class MapObject(ObjectInterface, MapView):
    init_resolution: float
    layers: List[LayerObject]


# noinspection PyAbstractClass
class ProjectObject(ObjectInterface):
    map: MapObject
    title: str
    locale: str
    meta: MetaData


class SearchArgs(Data):
    bbox: Extent
    crs: crsref
    keyword: Optional[str]
    layers: List[LayerObject]
    limit: int
    tolerance: int
    project: ProjectObject
    resolution: float
    shape: Optional[ShapeInterface]


class SearchProviderInterface(ObjectInterface):
    type: str
    feature_format: FormatInterface
    title: str
    keyword_required: bool
    geometry_required: bool

    def can_run(self, args: SearchArgs) -> bool:
        raise NotImplementedError

    def run(self, layer: LayerObject, args: SearchArgs) -> List[FeatureInterface]:
        raise NotImplementedError


class AuthProviderInterface:
    def authenticate_user(self, login: str, password: str, **kw) -> 'AuthUserInterface':
        raise NotImplementedError()

    def get_user(self, user_uid: str) -> 'AuthUserInterface':
        raise NotImplementedError()

    def unmarshal_user(self, user_uid: str, s: str) -> 'AuthUserInterface':
        raise NotImplementedError()

    def marshal_user(self, user: 'AuthUserInterface') -> str:
        raise NotImplementedError()


class AuthUserInterface:
    display_name: str
    props: dict
    is_guest: bool
    full_uid: str

    def init_from_source(self, provider: AuthProviderInterface, uid, roles=None, attributes=None):
        raise NotImplementedError

    def init_from_cache(self, provider: AuthProviderInterface, uid: str, roles: List[str], attributes: dict):
        raise NotImplementedError

    def attribute(self, key, default=''):
        raise NotImplementedError

    def can_read(self, obj: ObjectInterface) -> bool:
        raise NotImplementedError

    def can_write(self, obj: ObjectInterface) -> bool:
        raise NotImplementedError

    def can_execute(self, obj: ObjectInterface) -> bool:
        raise NotImplementedError

    def can(self, mode, obj: ObjectInterface) -> bool:
        raise NotImplementedError


class SqlTableConfig(Config):
    """SQL database table"""

    geometryColumn: Optional[str]  #: geometry column name
    keyColumn: Optional[str]  #: primary key column name
    name: str  #: table name
    searchColumn: Optional[str]  #: column to be searched for


class SelectArgs(Data):
    keyword: Optional[str]
    limit: Optional[int]
    tolerance: Optional[float]
    shape: Optional[ShapeInterface]
    sort: Optional[str]
    table: SqlTableConfig
    extraWhere: Optional[str]


# noinspection PyAbstractClass
class DbProviderObject(ObjectInterface):
    error: type
    connect_params: dict

    def connect(self):
        raise NotImplementedError

    def select(self, args: SelectArgs) -> List[FeatureInterface]:
        raise NotImplementedError

from typing import Optional, List, Dict, Tuple, Union, cast

import shapely.geometry

from .data import Data

#: alias: An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#: alias: Point coordinates [x, y]
Point = Tuple[float, float]

#: alias: Size [width, height]
Size = Tuple[float, float]

Config = Data


class Params(Data):
    projectUid: Optional[str]  #: project uid
    locale: Optional[str]  #: locale for this request


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

    class layer:
        class Config(Config):
            pass

    class search:
        class provider:
            class Config(Config):
                pass

    class storage:
        class Config(Config):
            pass

    class ows:
        class service:
            class Config(Config):
                pass


class filepath:
    """Valid readable file path on the server"""
    pass


class dirpath:
    """Valid readable directory path on the server"""
    pass


class duration:
    """String like "1w 2d 3h 4m 5s" or a number of seconds"""
    pass


class regex:
    """Regular expression, as used in Python"""
    pass


class formatstr:
    """String with {attribute} placeholders"""
    pass


class crsref:
    """CRS code like "EPSG:3857" """
    pass


class date:
    """ISO date like "2019-01-30" """
    pass


class url:
    """An http or https URL"""
    pass


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessRuleConfig(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a roles to which this rule applies


#: alias:
Access = List[AccessRuleConfig]


class ResponseError(Data):
    status: int
    info: str


class Response(Data):
    error: Optional[ResponseError]


class NoParams(Data):
    pass


class HttpResponse(Response):
    mimeType: str
    content: str
    status: int


class WithType(Config):
    type: str  #: object type


class WithTypeAndAccess(WithType):
    access: Optional[Access]  #: access rights


class DocumentRootConfig(Config):
    """Base directory for assets"""

    dir: dirpath  #: directory path
    allowMime: Optional[List[str]]  #: allowed mime types
    denyMime: Optional[List[str]]  #: disallowed mime types (from the standard list)


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
    """Contact metadata configuration"""

    address: str = ''
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
    url: url = ''


class MetaLink(Data):
    """Object link configuration"""

    scheme: str = ''  #: link scheme
    url: url  #: link url
    function: str = ''  #: ISO-19115 function, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#CI_OnLineFunctionCode


class MetaData(Data):
    """Object metadata configuration"""

    abstract: str = ''  #: object abstract description
    attribution: str = ''  #: attribution (copyright) string
    keywords: List[str] = []  #: keywords
    language: str = ''  #: object language
    name: str = ''  #: object internal name
    title: str = ''  #: object title

    accessConstraints: str = ''
    fees: str = ''

    # uid: str = ''  #: ISO-19115 identifier
    # category: str = ''  #: ISO-19115 category, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_TopicCategoryCode
    # scope: str = ''  #: ISO-19115 scope, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_ScopeCode
    iso: dict = {}  #: ISO-19115 properties

    # theme: str = ''  #: INSPIRE theme shortcut, e.g. "au"
    inspire: dict = {}  #: INSPIRE  properties

    contact: Optional[MetaContact]  #: contact information

    pubDate: date = ''  #: publication date
    modDate: date = ''  #: modification date

    image: url = ''  #: image (logo) url
    images: dict = {}  #: further images

    url: url = ''  #: object metadata url
    serviceUrl: url = ''  #: object service url
    links: List[MetaLink] = []  #: additional links


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
        self.resource_urls = {}

        self.a_path = ''
        self.a_uid = ''
        self.a_level = 0


#####

class AttributeType(Enum):
    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    float = 'float'
    geometry = 'geometry'
    int = 'int'
    list = 'list'
    str = 'str'
    time = 'time'

    geoCurve = 'curve'
    geoGeomcollection = 'geomcollection'
    geoGeometry = 'geometry'
    geoLinestring = 'linestring'
    geoMulticurve = 'multicurve'
    geoMultilinestring = 'multilinestring'
    geoMultipoint = 'multipoint'
    geoMultipolygon = 'multipolygon'
    geoMultisurface = 'multisurface'
    geoPoint = 'point'
    geoPolygon = 'polygon'
    geoPolyhedralsurface = 'polyhedralsurface'
    geoSurface = 'surface'


class AttributeConfig(Config):
    """Attribute configuration"""

    title: str = ''  #: title
    name: str = ''  #: internal name
    value: str = ''  #: computed value
    source: str = ''  #: source attribute
    type: Optional[AttributeType]  #: type


class Attribute(Data):
    title: str = ''
    name: str = ''
    type: str = ''
    value: str = ''


class ObjectInterface:
    children: List['ObjectInterface']
    config: object
    klass: str
    parent: 'ObjectInterface'
    root: 'ObjectInterface'
    uid: str
    props: Data

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

    def find_all(self, klass=None) -> 'ObjectInterface':
        raise NotImplementedError

    def find_first(self, klass) -> 'ObjectInterface':
        raise NotImplementedError

    def find(self, klass, uid) -> List['ObjectInterface']:
        raise NotImplementedError

    def props_for(self, user: 'AuthUserInterface') -> dict:
        raise NotImplementedError


class TemplateQualityLevel(Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class TemplateConfig(Config):
    type: str  #: template type
    qualityLevels: Optional[List[TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: Optional[List[AttributeConfig]]  #: user-editable template attributes
    path: Optional[filepath]  #: path to a template file
    text: str = ''  #: template content
    title: str = ''  #: template title
    uid: str = ''  #: unique id


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


class SvgFragment:
    points: List[Point]
    svg: str


class MapRenderInputItem(Data):
    bitmap: str
    features: List['FeatureInterface']
    layer: 'LayerObject'
    sub_layers: List[str]
    opacity: float
    print_as_vector: bool
    style: 'StyleProps'
    svg_fragment: dict


class MapRenderInput(Data):
    out_path: str
    bbox: Extent
    rotation: int
    scale: int
    dpi: int
    map_size_px: Size
    background_color: int
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


class MapView:
    crs: crsref
    extent: Extent
    resolutions: List[float]


# noinspection PyAbstractClass
class LayerObject(ObjectInterface, MapView):
    has_legend: bool
    has_cache: bool
    has_search: bool
    is_public: bool
    layers: List['LayerObject']

    map: 'MapObject'
    meta: MetaData
    opacity: float

    title: str
    description: str

    data_model: List[Attribute]

    def mapproxy_config(self, mc):
        raise NotImplementedError

    def render_bbox(self, bbox, width, height, **client_params):
        raise NotImplementedError

    def render_xyz(self, x, y, z):
        raise NotImplementedError

    def render_svg(self, bbox, dpi, scale, rotation, style):
        raise NotImplementedError

    def render_legend(self):
        raise NotImplementedError

    def get_features(self, bbox, limit):
        raise NotImplementedError

    def edit_access(self, user: 'AuthUserInterface'):
        raise NotImplementedError

    def add_features(self, features: List['FeatureProps']):
        raise NotImplementedError

    def update_features(self, features: List['FeatureProps']):
        raise NotImplementedError

    def delete_features(self, features: List['FeatureProps']):
        raise NotImplementedError

    def search(self, provider: 'SearchProviderInterface', args: 'SearchArgs') -> List['FeatureInterface']:
        raise NotImplementedError

    def ows_enabled(self, service: 'OwsServiceInterface') -> bool:
        raise NotImplementedError


class ShapeProps(Data):
    geometry: dict
    crs: str


class StyleProps(Data):
    """Feature style"""

    type: str  #: style type ("css")
    content: Optional[dict]  #: css rules
    text: Optional[str]  #: raw style content


class FeatureProps(Data):
    attributes: Optional[dict]
    category: str = ''
    description: str = ''
    label: str = ''
    shape: Optional[ShapeProps]
    style: Optional[StyleProps]
    teaser: str = ''
    title: str = ''
    uid: Optional[str]


class FormatConfig(Config):
    """Feature format"""

    description: Optional[ext.template.Config]  #: template for feature descriptions
    category: Optional[ext.template.Config]  #: feature category
    label: Optional[ext.template.Config]  #: feature label on the map
    dataModel: Optional[List[AttributeConfig]]  #: attribute metadata
    teaser: Optional[ext.template.Config]  #: template for feature teasers (short descriptions)
    title: Optional[ext.template.Config]  #: feature title


class FormatInterface(ObjectInterface):
    category: TemplateObject
    description: TemplateObject
    label: TemplateObject
    teaser: TemplateObject
    title: TemplateObject

    data_model: List[Attribute]

    def apply(self, feature: 'FeatureInterface', context: dict = None):
        """Format a feature."""
        raise NotImplementedError

    def apply_data_model(self, d: dict, data_model: List[Attribute]):
        """Convert data."""
        raise NotImplementedError


class OwsServiceInterface:
    name: str
    type: str
    meta: MetaData
    version: str


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

    def apply_format(self, fmt: FormatInterface, context: dict = None):
        raise NotImplementedError


# noinspection PyAbstractClass
class MapObject(ObjectInterface, MapView):
    init_resolution: float
    layers: List[LayerObject]
    coordinatePrecision: int


# noinspection PyAbstractClass
class ProjectObject(ObjectInterface):
    map: MapObject
    title: str
    locales: List[str]
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
    shapes: List[ShapeInterface]
    feature_format: FormatInterface


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

    def can_use(self, obj, parent=None) -> bool:
        raise NotImplementedError


class SqlTableConfig(Config):
    """SQL database table"""

    geometryColumn: Optional[str]  #: geometry column name
    keyColumn: str = 'id'  #: primary key column name
    name: str  #: table name
    searchColumn: Optional[str]  #: column to be searched for


class SelectArgs(Data):
    keyword: Optional[str]
    limit: Optional[int]
    tolerance: Optional[float]
    shape: Optional[ShapeInterface]
    sort: Optional[str]
    table: SqlTableConfig
    ids: Optional[List[str]]
    extraWhere: Optional[str]


# noinspection PyAbstractClass
class DbProviderObject(ObjectInterface):
    error: type
    connect_params: dict

    def connect(self, extra_connect_params: dict = None):
        raise NotImplementedError

    def select(self, args: SelectArgs, extra_connect_params: dict = None) -> List[FeatureInterface]:
        raise NotImplementedError

    def insert(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        raise NotImplementedError

    def update(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        raise NotImplementedError

    def delete(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        raise NotImplementedError

    def describe(self, table: SqlTableConfig) -> List[Attribute]:
        raise NotImplementedError


class StorageEntry(Data):
    category: str
    name: str


class StorageInterface(ObjectInterface):
    def read(self, entry: StorageEntry, user: AuthUserInterface) -> dict:
        raise NotImplementedError

    def write(self, entry: StorageEntry, user: AuthUserInterface, data: dict) -> str:
        raise NotImplementedError

    def dir(self, user: AuthUserInterface) -> List[StorageEntry]:
        raise NotImplementedError

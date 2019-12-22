# noinspection PyUnresolvedReferences
from typing import Optional, List, Dict, Tuple, Union, cast

# type: ignore

### Basic types

# noinspection PyUnresolvedReferences



# NB: we cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum:
    pass


#: alias: An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#: alias: Point coordinates [x, y]
Point = Tuple[float, float]

#: alias: Size [width, height]
Size = Tuple[float, float]


class Axis(Enum):
    xy = 'xy'
    yx = 'yx'


### semantic primitive types

class Literal(str):
    pass


class FilePath(str):
    """Valid readable file path on the server"""
    pass


class DirPath(str):
    """Valid readable directory path on the server"""
    pass


class Duration(str):
    """String like "1w 2d 3h 4m 5s" or a number of seconds"""
    pass


class Regex(str):
    """Regular expression, as used in Python"""
    pass


class FormatStr(str):
    """String with {attribute} placeholders"""
    pass


class Crs(str):
    """CRS code like "EPSG:3857" """
    pass


class Date(str):
    """ISO date like "2019-01-30" """
    pass


class Url(str):
    """An http or https URL"""
    pass

# type: ignore

### Data objects

class Data:
    """Data object."""

    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                setattr(self, str(k), v)

    def set(self, k, value):
        return setattr(self, k, value)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def as_dict(self):
        return vars(self)

    def extend(self, *args):
        d = {}
        for a in args:
            if not isinstance(a, dict):
                a = a.as_dict() if hasattr(a, 'as_dict') else None
            if a:
                d.update(a)
        vars(self).update(d)
        return self

    def __repr__(self):
        return repr(vars(self))


class Config(Data):
    """Configuration base type"""
    pass


class Props(Data):
    """Properties base type"""
    pass

### Basic tree node object.







class Object:
    children: List['Object']
    config: Config
    klass: str
    parent: 'Object'
    root: 'Object'
    uid: str
    props: Props

    def is_a(self, klass):
        pass

    def initialize(self, cfg):
        pass

    def configure(self):
        pass

    def var(self, key, default=None, parent=False):
        pass

    def add_child(self, klass, cfg):
        pass

    def get_children(self, klass) -> List['Object']:
        pass

    def get_closest(self, klass) -> 'Object':
        pass

    def find_all(self, klass=None) -> List['Object']:
        pass

    def find_first(self, klass) -> 'Object':
        pass

    def find(self, klass, uid) -> List['Object']:
        pass

    def props_for(self, user: 'AuthUser') -> Props:
        pass


class RootObject(Object):
    application: 'ApplicationObject'
    all_types: dict
    all_objects: List['Object']
    shared_objects: dict

    def create(self, klass, cfg=None) -> 'Object':
        pass

    def validate_action(self, category: str, cmd: str, payload: dict) -> Tuple[str, str, dict]:
        pass


### Access rules and configs.





class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessRuleConfig(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a roles to which this rule applies


#: alias:
Access = List[AccessRuleConfig]


class WithType(Config):
    type: str  #: object type


class WithTypeAndAccess(WithType):
    access: Optional[Access]  #: access rights

# type: ignore

### Attributes and data models.





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

    name: str = ''  #: internal name
    source: str = ''  #: source attribute
    title: str = ''  #: display title
    type: AttributeType = 'str'  #: type
    value: FormatStr = ''  #: computed value


class Attribute(Data):
    name: str
    title: str = ''
    type: str = 'str'
    value: str


class DataModel:
    attributes: List[Attribute]


class DataModelConfig(Config):
    """Data model configuration."""
    attributes: List[AttributeConfig]

# type: ignore

### Authorization provider and user






class AuthProviderObject(Object):
    def authenticate_user(self, login: str, password: str, **kw) -> 'AuthUser':
        pass

    def get_user(self, user_uid: str) -> 'AuthUser':
        pass

    def unmarshal_user(self, user_uid: str, s: str) -> 'AuthUser':
        pass

    def marshal_user(self, user: 'AuthUser') -> str:
        pass


class AuthUser:
    display_name: str
    props: Props
    is_guest: bool
    full_uid: str

    def init_from_source(self, provider: AuthProviderObject, uid: str, roles: List[str] = None, attributes: dict = None):
        pass

    def init_from_cache(self, provider: AuthProviderObject, uid: str, roles: List[str], attributes: dict):
        pass

    def attribute(self, key, default=''):
        pass

    def can_use(self, obj: 'Object', parent: 'Object' = None) -> bool:
        pass

class AuthUserProps(Props):
    displayName: str

# type: ignore

### Database-related.









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
    shape: Optional['Shape']
    sort: Optional[str]
    table: SqlTableConfig
    ids: Optional[List[str]]
    extraWhere: Optional[str]


class StorageEntry(Data):
    category: str
    name: str


class StorageObject(Object):
    def read(self, entry: StorageEntry, user: 'AuthUser') -> dict:
        return {}

    def write(self, entry: StorageEntry, user: 'AuthUser', data: dict) -> str:
        return ''

    def dir(self, user: 'AuthUser') -> List[StorageEntry]:
        return []


class SqlProviderObject(Object):
    error: type
    connect_params: dict

    def connect(self, extra_connect_params: dict = None):
        pass

    def select(self, args: SelectArgs, extra_connect_params: dict = None) -> List['Feature']:
        pass

    def insert(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def update(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def delete(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def describe(self, table: SqlTableConfig) -> List['Attribute']:
        pass
    
# type: ignore

### Dummy classes to support extension typing.


class ext:
    class action:
        class Config:
            pass

    class auth:
        class provider:
            class Config:
                pass

    class template:
        class Config:
            pass

        class Props:
            pass

    class db:
        class provider:
            class Config:
                pass

    class layer:
        class Config:
            pass

        class Props:
            pass

    class search:
        class provider:
            class Config:
                pass

    class storage:
        class Config:
            pass

    class ows:
        class provider:
            class Config:
                pass

        class service:
            class Config:
                pass

### Shapes and features.






import shapely.geometry.base

class ShapeProps(Props):
    geometry: dict
    crs: str



class Shape:
    crs: Crs
    geo: shapely.geometry.base.BaseGeometry
    props: dict
    type: str
    wkb: str
    wkb_hex: str
    wkt: str
    bounds: Extent

    def transform(self, to_crs):
        pass


class FeatureProps(Data):
    attributes: Optional[dict]
    category: str = ''
    description: str = ''
    label: str = ''
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']
    teaser: str = ''
    title: str = ''
    uid: Optional[str]


class Feature:
    attributes: dict
    description: str
    category: str
    label: str
    props: 'FeatureProps'
    shape: 'Shape'
    shape_props: 'ShapeProps'
    style: 'StyleProps'
    teaser: str
    title: str
    uid: str

    def transform(self, to_crs):
        """Transform the feature to another CRS"""
        pass

    def to_svg(self, bbox, dpi, scale, rotation):
        """Render the feature as SVG"""
        pass

    def to_geojson(self):
        """Render the feature as GeoJSON"""
        pass

    def apply_format(self, fmt: 'FormatObject', context: dict = None):
        pass
### Maps and layers












class LayerObject(Object):
    has_legend: bool
    has_cache: bool
    has_search: bool
    is_public: bool
    layers: List['LayerObject']

    map: 'MapObject'
    meta: 'MetaData'
    opacity: float

    title: str
    description: str

    crs: Crs
    extent: Extent
    resolutions: List[float]

    data_model: DataModel
    feature_format: 'FormatObject'


    def mapproxy_config(self, mc):
        pass

    def render_bbox(self, bbox, width, height, **client_params):
        pass

    def render_xyz(self, x, y, z):
        pass

    def render_svg(self, bbox, dpi, scale, rotation, style):
        pass

    def render_legend(self):
        pass

    def get_features(self, bbox, limit) -> List[Feature]:
        return []

    def edit_access(self, user: 'AuthUser'):
        pass

    def add_features(self, features: List['FeatureProps']):
        pass

    def update_features(self, features: List['FeatureProps']):
        pass

    def delete_features(self, features: List['FeatureProps']):
        pass

    def search(self, provider: 'SearchProviderObject', args: 'SearchArguments') -> List['SearchResult']:
        return []

    def ows_enabled(self, service: 'OwsServiceObject') -> bool:
        return False


class MapObject(Object):
    init_resolution: float
    layers: List['LayerObject']
    coordinatePrecision: int
    crs: Crs
    extent: Extent
    resolutions: List[float]
    coordinate_precision: int

class ProjectObject(Object):
    map: MapObject
    title: str
    locales: List[str]
    meta: 'MetaData'

### Metadata.




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
    url: Url = ''


class MetaLink(Data):
    """Object link configuration"""

    scheme: str = ''  #: link scheme
    url: Url  #: link url
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

    pubDate: Date = ''  #: publication date
    modDate: Date = ''  #: modification date

    image: Url = ''  #: image (logo) url
    images: dict = {}  #: further images

    url: Url = ''  #: object metadata url
    serviceUrl: Url = ''  #: object service url
    links: List[MetaLink] = []  #: additional links


### Miscellaneous types.




class DocumentRootConfig(Config):
    """Base directory for assets"""

    dir: DirPath  #: directory path
    allowMime: Optional[List[str]]  #: allowed mime types
    denyMime: Optional[List[str]]  #: disallowed mime types (from the standard list)



### OWS providers and services.








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

### Request params and responses.







import werkzeug.wrappers


class Params(Data):
    projectUid: Optional[str]  #: project uid
    locale: Optional[str]  #: locale for this request


class NoParams(Data):
    pass


class ResponseError(Data):
    status: int
    info: str


class Response(Data):
    error: Optional[ResponseError]


class HttpResponse(Response):
    mimeType: str
    content: str
    status: int


class Request:
    environ: dict
    cookies: dict
    has_struct: bool
    expected_struct: str
    data: bytes
    params: dict
    kparams: dict
    post_data: str
    user: 'AuthUser'

    def response(self, content, mimetype: str, status=200) -> werkzeug.wrappers.Response:
        pass

    def struct_response(self, data, status=200) -> werkzeug.wrappers.Response:
        pass

    def env(self, key: str, default=None) -> str:
        pass

    def param(self, key: str, default=None) -> str:
        pass

    def kparam(self, key: str, default=None) -> str:
        pass

    def url_for(self, url: str) -> str:
        pass

    def require(self, klass: str, uid: str) -> Object:
        pass

    def require_project(self, uid: str) -> 'ProjectObject':
        pass

    def acquire(self, klass: str, uid: str) -> Object:
        pass

    def login(self, username: str, password: str):
        pass

    def logout(self):
        pass

    def auth_begin(self):
        pass

    def auth_commit(self, res: werkzeug.wrappers.Response):
        pass

### Search









class SearchArguments(Data):
    axis: str
    bbox: Extent
    count: int
    crs: Crs
    feature_format: 'FormatObject'
    keyword: Optional[str]
    layers: List['LayerObject']
    limit: int
    params: dict
    point: Point
    project: 'ProjectObject'
    resolution: float
    shapes: List['Shape']
    tolerance: int


class SearchResult(Data):
    feature: 'Feature'
    layer: 'LayerObject'
    provider: 'SearchProviderObject'


class SearchProviderObject(Object):
    feature_format: 'FormatObject'
    geometry_required: bool
    keyword_required: bool
    title: str
    type: str

    def can_run(self, args: SearchArguments) -> bool:
        return False

    def run(self, layer: Optional['LayerObject'], args: SearchArguments) -> List['Feature']:
        return []

    def context_shape(self, args: SearchArguments):
        pass

### Styles





class StyleProps(Props):
    type: str
    content: Optional[dict]
    text: Optional[str]


class StyleConfig(Config):
    """Feature style"""

    type: str  #: style type ("css")
    content: Optional[dict]  #: css rules
    text: Optional[str]  #: raw style content

### Templates, renderers and formats.











class TemplateQualityLevel(Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class TemplateConfig(Config):
    type: str  #: template type
    qualityLevels: Optional[List[TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: Optional[DataModelConfig]  #: user-editable template attributes
    path: Optional[FilePath]  #: path to a template file
    text: str = ''  #: template content
    title: str = ''  #: template title
    uid: str = ''  #: unique id


class TemplateProps(Props):
    uid: str
    title: str
    qualityLevels: List[TemplateQualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: DataModel


class TemplateRenderOutput(Data):
    mimeType: str
    content: str
    path: str


class SvgFragment:
    points: List[Point]
    svg: str


class MapRenderInputItem(Data):
    bitmap: str
    features: List['Feature']
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


class TemplateObject(Object):
    data_model: DataModel
    map_size: List[int]
    page_size: List[int]

    def dpi_for_quality(self, quality: int) -> int:
        pass

    def render(self, context: dict, render_output: MapRenderOutput = None,
               out_path: str = None, format: str = None) -> TemplateRenderOutput:
        pass

    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str:
        pass

    def normalize_user_data(self, data: dict) -> dict:
        pass


class FormatConfig(Config):
    """Feature format"""

    description: Optional[ext.template.Config]  #: template for feature descriptions
    category: Optional[ext.template.Config]  #: feature category
    label: Optional[ext.template.Config]  #: feature label on the map
    teaser: Optional[ext.template.Config]  #: template for feature teasers (short descriptions)
    title: Optional[ext.template.Config]  #: feature title


class FormatObject(Object):
    category: TemplateObject
    description: TemplateObject
    label: TemplateObject
    teaser: TemplateObject
    title: TemplateObject

    def apply(self, feature: 'Feature', context: dict = None):
        """Format a feature."""
        pass

### Application









class ApiObject(Object):
    actions: dict


class ClientObject(Object):
    pass


class CorsConfig(Config):
    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: Optional[List[str]]


class RewriteRule(Config):
    match: Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: Optional[dict]  #: additional options


class WebSiteObject(Object):
    host: str
    ssl: bool
    error_page: 'TemplateObject'
    static_root: 'DocumentRootConfig'
    assets_root: 'DocumentRootConfig'
    rewrite_rules: List[RewriteRule]
    reversed_rewrite_rules: List[RewriteRule]
    cors: CorsConfig

    def url_for(self, req, url: str) -> str:
        pass


class ApplicationObject(Object):
    api: ApiObject
    client: ClientObject
    qgis_version: str
    storage: 'StorageObject'
    version: str
    web_sites: List[WebSiteObject]

    def find_action(self, action_type: str, project_uid=None) -> Object:
        pass


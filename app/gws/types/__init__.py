# noinspection PyUnresolvedReferences
from typing import Any, Dict, List, Optional, Tuple, Union, cast
# noinspection PyUnresolvedReferences
from .data import Data, Config, Props

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


class Attribute(Data):
    name: str
    title: str = ''
    type: AttributeType = 'str'
    value: Any = None


class ModelRule(Data):
    """Attribute conversion rule"""

    name: str = ''  #: target attribute name
    value: Optional[str]  #: constant value
    source: str = ''  #: source attribute
    title: str = ''  #: target attribute display title
    type: AttributeType = 'str'  #: target attribute type
    format: FormatStr = ''  #: attribute formatter
    expression: str = ''  #: attribute formatter


class ModelConfig(Config):
    """Data model."""
    rules: List[ModelRule]


class ModelProps(Props):
    rules: List[ModelRule]


### Database-related.







class SqlTableConfig(Config):
    """SQL database table"""

    name: str  #: table name
    geometryColumn: Optional[str]  #: geometry column name
    keyColumn: Optional[str]  #: primary key column name
    searchColumn: Optional[str]  #: column to be searched for


class SqlTable(Data):
    name: str
    key_column: str = ''
    search_column: str = ''
    geometry_column: str = ''
    geometry_type: str = ''
    geometry_crs: Crs = ''


class SelectArgs(Data):
    keyword: Optional[str]
    limit: Optional[int]
    tolerance: Optional[float]
    shape: Optional['Shape']
    sort: Optional[str]
    table: SqlTable
    uids: Optional[List[str]]
    extraWhere: Optional[str]


class SqlTableColumn(Data):
    name: str
    type: 'AttributeType'
    native_type: str
    crs: Crs
    is_key: bool
    is_geometry: bool


#: alias:
SqlTableDescription = Dict[str, SqlTableColumn]


class StorageEntry(Data):
    category: str
    name: str

### Shapes and features.






import shapely.geometry.base


class ShapeProps(Props):
    geometry: dict
    crs: str


class FeatureProps(Data):
    uid: str = ''
    attributes: List['Attribute'] = ''
    elements: dict = {}
    layerUid: str = ''
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']

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




class OwsOperation:
    def __init__(self):
        self.name = ''
        self.formats: List[str] = []
        self.get_url: Url = ''
        self.post_url: Url = ''
        self.parameters: dict = {}

### Request params and responses.





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
    mime: str
    content: str
    status: int


class FileResponse(Response):
    mime: str
    path: str
    status: int
    attachment_name: str

### Search





class SearchArgs(Data):
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

## Map renderer





import PIL.Image


class SvgFragment:
    points: List[Point]
    svg: str


class RenderView(Data):
    bbox: Extent
    center: Point
    dpi: int
    rotation: int
    scale: int
    size_mm: Size
    size_px: Size


class RenderInputItemType(Enum):
    image = 'image'
    features = 'features'
    fragment = 'fragment'
    svg_layer = 'svg_layer'
    bbox_layer = 'bbox_layer'


class RenderInputItem(Data):
    type: str = ''
    image: PIL.Image.Image = None
    features: List['Feature']
    layer: 'LayerObject' = None
    sub_layers: List[str] = []
    opacity: float = None
    print_as_vector: bool = None
    style: 'Style' = None
    fragment: 'SvgFragment' = None
    dpi: int = None


class RenderInput(Data):
    view: 'RenderView'
    background_color: int
    items: List[RenderInputItem]


class RenderOutputItemType(Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


class RenderOutputItem(Data):
    type: str
    image: PIL.Image.Image
    path: str = ''
    elements: List[str] = []


class RenderOutput(Data):
    view: 'RenderView'
    items: List[RenderOutputItem]

### Templates and formats.









class TemplateQualityLevel(Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class TemplateConfig(Config):
    type: str  #: template type
    qualityLevels: Optional[List[TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: Optional[ModelConfig]  #: user-editable template attributes
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
    dataModel: 'ModelProps'


class TemplateOutput(Data):
    mime: str
    content: str
    path: str


class FeatureFormatConfig(Config):
    """Feature format"""

    description: Optional[ext.template.Config]  #: template for feature descriptions
    category: Optional[ext.template.Config]  #: feature category
    label: Optional[ext.template.Config]  #: feature label on the map
    teaser: Optional[ext.template.Config]  #: template for feature teasers (short descriptions)
    title: Optional[ext.template.Config]  #: feature title


class LayerFormatConfig(Config):
    """Layer format"""

    description: Optional[ext.template.Config]  #: template for the layer description



### Application





class CorsConfig(Config):
    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: Optional[List[str]]


class RewriteRule(Config):
    match: Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: Optional[dict]  #: additional options

class BaseWebRequest:
    cookies : dict
    data : Optional[bytes]
    environ : dict
    input_struct_type : int
    kparams : dict
    output_struct_type : int
    params : dict
    text_data : Optional[str]
    def env(self, key: str, default: str = None) -> str: pass
    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> 'WebResponse': pass
    def kparam(self, key: str, default: str = None) -> str: pass
    def param(self, key: str, default: str = None) -> str: pass
    def response(self, content: str, mimetype: str, status: int = 200) -> 'WebResponse': pass
    def struct_response(self, data: 'Response', status: int = 200) -> 'WebResponse': pass
    def url_for(self, url: 'Url') -> 'Url': pass

class Feature:
    attributes : List[Attribute]
    convertor : 'FeatureConvertor'
    elements : dict
    layer : 'LayerObject'
    props : 'FeatureProps'
    shape : 'Shape'
    style : 'Style'
    uid : str
    def convert(self, target_crs: 'Crs' = None, convertor: 'FeatureConvertor' = None) -> 'Feature': pass
    def set_default_style(self, style): pass
    def to_geojson(self): pass
    def to_svg(self, rv: 'RenderView', style: 'Style' = None): pass
    def transform(self, to_crs): pass

class FeatureConvertor:
    data_model : 'ModelObject'
    feature_format : 'FormatObject'

class Object:
    auto_uid : str
    children : list
    props : Props
    root : 'RootObject'
    uid : str
    def add_child(self, klass, cfg): pass
    def configure(self): pass
    def create_object(self, klass, cfg, parent=None): pass
    def create_shared_object(self, klass, uid, cfg): pass
    def find(self, klass, uid): pass
    def find_all(self, klass=None): pass
    def find_first(self, klass): pass
    def get_children(self, klass): pass
    def get_closest(self, klass): pass
    def initialize(self, cfg): pass
    def is_a(self, klass): pass
    def props_for(self, user): pass
    def set_uid(self, uid): pass
    def var(self, key, default=None, parent=False): pass

class Role:
    def can_use(self, obj, parent=None): pass

class Shape:
    bounds : 'Extent'
    crs : 'Crs'
    crs_code : str
    geo : shapely.geometry.base.BaseGeometry
    props : 'ShapeProps'
    type : str
    wkb : str
    wkb_hex : str
    wkt : str
    def tolerance_buffer(self, tolerance, resolution=None) -> 'Shape': pass
    def transform(self, to_crs) -> 'Shape': pass

class SourceLayer:
    a_level : int
    a_path : str
    a_uid : str
    data_source : dict
    extents : Dict['Crs', 'Extent']
    is_expanded : bool
    is_group : bool
    is_image : bool
    is_queryable : bool
    is_visible : bool
    layers : List['SourceLayer']
    legend : str
    meta : 'MetaData'
    name : str
    opacity : int
    resource_urls : dict
    scale_range : List[float]
    styles : List['SourceStyle']
    supported_crs : List['Crs']
    title : str

class SourceStyle:
    is_default : bool
    legend : 'Url'
    meta : 'MetaData'

class Style:
    pass

class User:
    attributes : dict
    display_name : str
    full_uid : str
    is_guest : bool
    provider : 'AuthProviderObject'
    roles : List[str]
    uid : str
    def attribute(self, key: str, default: str = '') -> str: pass
    def can_use(self, obj: Object, parent: Object = None) -> bool: pass
    def has_role(self, role: str) -> bool: pass
    def init_from_cache(self, provider, uid, roles, attributes) -> 'User': pass
    def init_from_source(self, provider, uid, roles=None, attributes=None) -> 'User': pass

class WebResponse:
    pass

class ApiObject(Object):
    actions : dict

class ApplicationObject(Object):
    api : 'ApiObject'
    client : 'ClientObject'
    qgis_version : str
    storage : 'StorageObject'
    web_sites : List['WebSiteObject']
    def find_action(self, action_type, project_uid=None): pass

class AuthProviderObject(Object):
    def get_user(self, user_uid: str) -> 'User': pass
    def marshal_user(self, user: 'User') -> str: pass
    def unmarshal_user(self, user_uid: str, json: str) -> 'User': pass

class ClientObject(Object):
    pass

class FormatObject(Object):
    templates : dict
    def apply(self, context: dict) -> dict: pass

class LayerObject(Object):
    cache : 'CacheConfig'
    can_render_bbox : bool
    can_render_svg : bool
    can_render_xyz : bool
    crs : str
    data_model : 'ModelObject'
    description_template : 'TemplateObject'
    display : str
    edit_data_model : 'ModelObject'
    edit_style : 'Style'
    extent : list
    feature_format : 'FormatObject'
    grid : 'GridConfig'
    has_cache : bool
    has_legend : bool
    image_format : str
    is_public : bool
    layers : list
    resolutions : list
    services : list
    style : 'Style'
    def edit_access(self, user): pass
    def mapproxy_config(self, mc): pass
    def ows_enabled(self, service): pass
    def render_bbox(self, rv: 'RenderView', client_params=None): pass
    def render_legend(self): pass
    def render_svg(self, rv: 'RenderView', style=None): pass
    def render_xyz(self, x, y, z): pass
    def use_meta(self, meta): pass

class MapObject(Object):
    center : list
    coordinate_precision : int
    crs : str
    extent : list
    init_resolution : int
    layers : List['LayerObject']
    resolutions : list

class ModelObject(Object):
    rules : List['ModelRule']
    def apply(self, atts: List[Attribute]) -> List[Attribute]: pass
    def apply_to_dict(self, d: dict) -> List[Attribute]: pass

class OwsProviderObject(Object):
    meta : 'MetaData'
    operations : List['OwsOperation']
    source_layers : List['SourceLayer']
    supported_crs : List['Crs']
    type : str
    url : 'Url'
    version : str
    def find_features(self, args: 'SearchArgs') -> List['Feature']: pass
    def operation(self, name: str) -> 'OwsOperation': pass

class OwsServiceObject(Object):
    feature_namespace : str
    local_namespaces : dict
    name : str
    templates : dict
    type : str
    use_inspire_data : bool
    use_inspire_meta : bool
    version : str
    def configure_inspire_templates(self): pass
    def configure_template(self, name, path, type='xml'): pass
    def dispatch(self, rd: 'RequestData', request_param): pass
    def error_response(self, status): pass
    def handle(self, req) -> 'HttpResponse': pass
    def is_layer_enabled(self, layer): pass
    def render_feature_nodes(self, rd: 'RequestData', nodes, container_template_name): pass
    def render_template(self, rd: 'RequestData', template, context, format=None): pass

class PrinterObject(Object):
    templates : List['TemplateObject']

class ProjectObject(Object):
    api : 'ApiObject'
    assets_root : 'DocumentRootConfig'
    client : 'ClientObject'
    description_template : 'TemplateObject'
    locales : list
    map : 'MapObject'
    meta : 'MetaData'
    overview_map : 'MapObject'
    printer : 'PrinterObject'
    title : str

class RootBase(Object):
    all_objects : list
    all_types : dict
    shared_objects : dict
    def create(self, klass, cfg=None): pass

class SearchProviderObject(Object):
    geometry_required : bool
    keyword_required : bool
    def can_run(self, args: 'SearchArgs'): pass
    def context_shape(self, args: 'SearchArgs'): pass

class SqlProviderObject(Object):
    def describe(self, table: 'SqlTable') -> 'SqlTableDescription': pass
    def edit_operation(self, operation: str, table: 'SqlTable', features: List['Feature']) -> List['Feature']: pass
    def select(self, args: 'SelectArgs', extra_connect_params: dict = None) -> List['Feature']: pass

class StorageObject(Object):
    def can_read(self, r, user): pass
    def can_write(self, r, user): pass
    def dir(self, user): pass
    def read(self, entry, user): pass
    def write(self, entry, user, data): pass

class TemplateObject(Object):
    data_model : 'ModelObject'
    map_size : 'Size'
    page_size : 'Size'
    def dpi_for_quality(self, quality): pass
    def normalize_user_data(self, d: dict) -> List[Attribute]: pass

class WebRequest(BaseWebRequest):
    user : 'User'
    def acquire(self, klass: str, uid: str) -> Object: pass
    def auth_begin(self): pass
    def auth_commit(self, res): pass
    def login(self, username: str, password: str): pass
    def logout(self): pass
    def require(self, klass: str, uid: str) -> Object: pass
    def require_project(self, uid: str) -> 'ProjectObject': pass

class WebSiteObject(Object):
    assets_root : 'DocumentRootConfig'
    cors : 'CorsConfig'
    error_page : 'TemplateObject'
    host : str
    reversed_rewrite_rules : list
    rewrite_rules : list
    ssl : bool
    static_root : 'DocumentRootConfig'
    def url_for(self, req, url): pass

class RootObject(RootBase):
    application : 'ApplicationObject'
    def configure(self): pass
    def validate_action(self, category, cmd, payload): pass
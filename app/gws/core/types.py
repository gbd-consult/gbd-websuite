from gws.types import Any, Dict, Enum, List, Literal, Optional, Protocol, Set, Tuple, Union
from .data import Data

# ----------------------------------------------------------------------------------------------------------------------
# custom types, used everywhere


#: An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#: Point coordinates [x, y]
Point = Tuple[float, float]

#: Size [width, height]
Size = Tuple[float, float]

#: A value with a unit
Measurement = Tuple[float, str]

#: An XML generator tag
Tag = tuple

#: Axis orientation
Axis = Literal['xy', 'yx']

#: Valid readable file path on the server
FilePath = str

#: Valid readable directory path on the server
DirPath = str

#: String like "1w 2d 3h 4m 5s" or a number of seconds
Duration = str

#: CSS color name
Color = str

#: Regular expression, as used in Python
Regex = str

#: String with {attribute} placeholders
FormatStr = str

#: CRS code like "EPSG:3857
Crs = str

#: ISO date like "2019-01-30"
Date = str

#: ISO date/time like "2019-01-30 01:02:03"
DateTime = str

#: Http or https URL
Url = str


class Bounds(Data):
    crs: Crs
    extent: Extent


# ----------------------------------------------------------------------------------------------------------------------
# application manifest


class ManifestPlugin(Data):
    path: FilePath
    name: str = ''


class Manifest(Data):
    addPlugins: Optional[List[ManifestPlugin]]
    excludePlugins: Optional[List[str]]
    plugins: Optional[List[ManifestPlugin]]

    withFallbackConfig: bool = False
    withStrictConfig: bool = False


# ----------------------------------------------------------------------------------------------------------------------
# basic objects

Klass = Union[type, str]


class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class Access(Data):
    """Access rights definition for authorization roles"""

    type: Literal['allow', 'deny']  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithAccess(Config):
    access: Optional[List[Access]]  #: access rights


class Props(Data):
    """Properties base type"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# foundation interfaces

class IObject(Protocol):
    class_name: str
    access: Optional[List[Access]]

    def props_for(self, user: 'IGrantee') -> Optional[Props]: ...

    def access_for(self, user: 'IGrantee') -> Optional[bool]: ...

    def is_a(self, klass: Klass) -> bool: ...


class IGrantee(Protocol):
    roles: Set[str]

    def can_use(self, obj: 'IObject', parent: 'IObject' = None) -> bool: ...


class INode(IObject, Protocol):
    children: List['INode']
    ext_category: str
    ext_type: str
    parent: Optional['INode']
    root: 'IRoot'
    title: str
    uid: str

    def configure(self): ...

    def post_configure(self): ...

    def var(self, key: str, default=None, with_parent: bool = False): ...

    def create_child(self, klass: Klass, cfg: Optional[Any]): ...

    def create_children(self, klass: Klass, cfg: Optional[Any]) -> List: ...

    def create_child_if_config(self, klass: Klass, cfg: Optional[Any]): ...

    def require_child(self, klass: Klass, cfg: Optional[Any]): ...

    def get_closest(self, klass: Klass): ...


class IRoot(Protocol):
    application: 'IApplication'
    specs: 'ISpecRuntime'
    configuration_errors: List[str]

    def set_object_uid(self, obj: 'INode', uid=None): ...

    def find_all(self, klass: Klass = None, uid: str = None) -> List: ...

    def find(self, klass: Klass = None, uid: str = None): ...

    def create_object(self, klass, cfg=None, parent: 'INode' = None, shared: bool = False, key=None): ...


# ----------------------------------------------------------------------------------------------------------------------
# spec runtime


class ExtObjectDescriptor(Data):
    class_ptr: type
    ext_category: str
    ext_type: str
    ident: str
    module_name: str
    module_path: str
    name: str


class ExtCommandDescriptor(Data):
    class_name: str
    cmd_action: str
    cmd_name: str
    function_name: str
    params: 'Params'


class ISpecRuntime(Protocol):
    manifest: Manifest

    def parse_command(self, cmd: str, method: str, params, with_strict_mode=True) -> Optional[ExtCommandDescriptor]: ...

    def read_value(self, value, type_name: str, path='', with_strict_mode=True, with_error_details=True, with_internal_objects=False): ...

    def real_class_names(self, class_name: str) -> List[str]: ...

    def object_descriptor(self, class_name: str) -> Optional[ExtObjectDescriptor]: ...

    def bundle_paths(self, category: str) -> List[str]: ...

    def cli_docs(self, lang) -> Dict: ...

    def is_a(self, class_name: str, partial_name: str) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# requests and responses

class Params(Data):
    """Web request params"""

    projectUid: Optional[str]  #: project uid
    localeUid: Optional[str]  #: locale for this request


class NoParams(Data):
    """Empty web request params"""
    pass


class ResponseError(Data):
    """Web response error"""
    status: int
    info: str


class Response(Data):
    """Web response base type"""
    error: Optional[ResponseError]


class ContentResponse(Response):
    """Web response with literal content"""

    as_attachment: bool
    attachment_name: str
    content: Union[bytes, str]
    location: str
    mime: str
    path: str
    status: int


class BytesResponse(Response):
    content: bytes
    mime: str


# noinspection PyPropertyDefinition
class IWebRequest(IObject, Protocol):
    site: 'IWebSite'

    @property
    def is_secure(self) -> bool: ...

    @property
    def user(self) -> 'IUser': ...

    def acquire(self, klass: str, uid: Optional[str]): ...

    def require(self, klass: str, uid: Optional[str]): ...

    def require_layer(self, uid: Optional[str]) -> 'ILayer': ...

    def require_project(self, uid: Optional[str]) -> 'IProject': ...

    def has_param(self, key: str) -> bool: ...

    def param(self, key: str, default: str = '') -> str: ...

    def cookie(self, key: str, default: str = '') -> str: ...

    def header(self, key: str, default: str = '') -> str: ...

    def env(self, key: str, default: str = '') -> str: ...


# noinspection PyPropertyDefinition
class IWebResponse(IObject, Protocol):
    @property
    def status_code(self) -> int: ...

    def set_cookie(self, key: str, **kwargs): ...

    def delete_cookie(self, key: str, **kwargs): ...

    def add_header(self, key: str, value): ...


# ----------------------------------------------------------------------------------------------------------------------
# web sites


class DocumentRoot(Data):
    dir: DirPath
    allow_mime: Optional[List[str]]
    deny_mime: Optional[List[str]]


class IWebSite(INode, Protocol):
    assets_root: Optional['DocumentRoot']

    def url_for(self, req: 'IWebRequest', url: Url) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization

class IAuthManager(INode, Protocol):
    guest_user: 'IUser'
    providers: List['IAuthProvider']
    methods: List['IAuthMethod']

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def get_user(self, user_uid: str) -> Optional['IUser']: ...

    def get_provider(self, uid: str = None, ext_type: str = None) -> Optional['IAuthProvider']: ...

    def get_method(self, uid: str = None, ext_type: str = None) -> Optional['IAuthMethod']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...

    def open_session(self, req: 'IWebRequest') -> 'IAuthSession': ...

    def close_session(self, sess: 'IAuthSession', req: 'IWebRequest', res: 'IWebResponse') -> 'IAuthSession': ...


class IAuthMethod(INode, Protocol):
    secure: bool

    def open_session(self, auth: 'IAuthManager', req: 'IWebRequest') -> Optional['IAuthSession']: ...

    def close_session(self, auth: IAuthManager, sess: 'IAuthSession', req: IWebRequest, res: IWebResponse): ...

    def login(self, auth: IAuthManager, credentials: Data, req: IWebRequest) -> Optional['IAuthSession']: ...

    def logout(self, auth: IAuthManager, sess: 'IAuthSession', req: IWebRequest) -> 'IAuthSession': ...


class IAuthProvider(INode, Protocol):
    allowed_methods: List[str]

    def get_user(self, local_uid: str) -> Optional['IUser']: ...

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...


class IAuthSession(IObject, Protocol):
    changed: bool
    data: dict
    method: Optional['IAuthMethod']
    typ: str
    uid: str
    user: 'IUser'

    def get(self, key: str, default=None): ...

    def set(self, key: str, val: Any): ...


# noinspection PyPropertyDefinition
class IUser(IObject, IGrantee, Protocol):
    attributes: Dict[str, Any]
    display_name: str
    is_guest: bool
    local_uid: str
    name: str
    provider: 'IAuthProvider'
    uid: str


# ----------------------------------------------------------------------------------------------------------------------
# attributes and models

class AttributeType(Enum):
    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    float = 'float'
    floatlist = 'floatlist'
    geometry = 'geometry'
    int = 'int'
    intlist = 'intlist'
    str = 'str'
    strlist = 'strlist'
    text = 'text'
    time = 'time'


class GeometryType(Enum):
    curve = 'CURVE'
    geomcollection = 'GEOMCOLLECTION'
    geometry = 'GEOMETRY'
    linestring = 'LINESTRING'
    multicurve = 'MULTICURVE'
    multilinestring = 'MULTILINESTRING'
    multipoint = 'MULTIPOINT'
    multipolygon = 'MULTIPOLYGON'
    multisurface = 'MULTISURFACE'
    point = 'POINT'
    polygon = 'POLYGON'
    polyhedralsurface = 'POLYHEDRALSURFACE'
    surface = 'SURFACE'


class Attribute(Data):
    name: str
    title: str = ''
    type: AttributeType = AttributeType.str
    value: Optional[Any]
    editable: bool = True


class IDataModel(INode, Protocol):
    def apply(self, attributes: List['Attribute']) -> List['Attribute']: ...

    def apply_to_dict(self, attr_values: dict) -> List['Attribute']: ...


# ----------------------------------------------------------------------------------------------------------------------
# shapes and features

# noinspection PyPropertyDefinition
class IShape(IObject, Protocol):
    crs: str
    srid: int

    @property
    def area(self) -> float: ...

    @property
    def bounds(self) -> Bounds: ...

    @property
    def centroid(self) -> 'IShape': ...

    @property
    def ewkb(self) -> bytes: ...

    @property
    def ewkb_hex(self) -> str: ...

    @property
    def ewkt(self) -> str: ...

    @property
    def extent(self) -> Extent: ...

    @property
    def geometry_type(self) -> 'GeometryType': ...

    @property
    def wkb(self) -> bytes: ...

    @property
    def wkb_hex(self) -> str: ...

    @property
    def wkt(self) -> str: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    def intersects(self, shape: 'IShape') -> bool: ...

    def to_multi(self) -> 'IShape': ...

    def to_type(self, new_type: 'GeometryType') -> 'IShape': ...

    def to_geojson(self) -> str: ...

    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape': ...

    def transformed_to(self, to_crs, **kwargs) -> 'IShape': ...


# noinspection PyPropertyDefinition
class IFeature(IObject, Protocol):
    attributes: List['Attribute']
    category: str
    data_model: Optional['IDataModel']
    elements: dict
    layer: Optional['ILayer']
    shape: Optional['IShape']
    style: Optional['IStyle']
    templates: Optional['ITemplateBundle']
    uid: str

    @property
    def template_context(self) -> dict: ...

    def apply_data_model(self, model: 'IDataModel' = None) -> 'IFeature': ...

    def apply_templates(self, templates: 'ITemplateBundle' = None, extra_context: dict = None, keys: List[str] = None) -> 'IFeature': ...

    def attr(self, name: str): ...

    def to_geojson(self) -> dict: ...

    def to_svg(self, rv: 'MapRenderView', style: 'IStyle' = None) -> str: ...

    def to_svg_tags(self, rv: 'MapRenderView', style: 'IStyle' = None) -> List['Tag']: ...

    def transform_to(self, crs: Crs) -> 'IFeature': ...

    def connect_to(self, layer: 'ILayer') -> 'IFeature': ...


# ----------------------------------------------------------------------------------------------------------------------
# database


class SqlTable(Data):
    name: str
    key_column: str
    search_column: str
    geometry_column: str
    geometry_type: GeometryType
    geometry_crs: Crs


class SqlSelectArgs(Data):
    extra_where: Optional[list]
    keyword: Optional[str]
    limit: Optional[int]
    map_tolerance: Optional[float]
    shape: Optional['IShape']
    sort: Optional[str]
    table: SqlTable
    uids: Optional[List[str]]
    columns: Optional[List[str]]


class SqlTableColumn(Data):
    name: str
    type: AttributeType
    geom_type: GeometryType
    native_type: str
    crs: Crs
    is_key: bool
    is_geometry: bool


class IDbProvider(INode, Protocol):
    pass


class ISqlDbProvider(IDbProvider, Protocol):
    def select(self, args: 'SqlSelectArgs', extra_connect_params: Optional[dict]) -> List['IFeature']: ...

    def insert(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def update(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def delete(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def describe(self, table: 'SqlTable') -> Dict[str, 'SqlTableColumn']: ...


# ----------------------------------------------------------------------------------------------------------------------
# templates and rendering


class IImage(IObject, Protocol):
    pass


class SvgFragment(Data):
    points: List['Point']
    styles: Optional[List['IStyle']]
    tags: List['Tag']


class MapRenderInputItemType(Enum):
    features = 'features'
    fragment = 'fragment'
    image = 'image'
    image_layer = 'image_layer'
    svg_layer = 'svg_layer'


class MapRenderInputItem(Data):
    dpi: int
    features: List['IFeature']
    fragment: 'SvgFragment'
    layer: 'ILayer'
    opacity: float
    print_as_vector: bool
    style: 'IStyle'
    sub_layers: List[str]
    image: 'IImage'
    type: 'MapRenderInputItemType'


class MapRenderView(Data):
    bounds: Bounds
    center: Point
    dpi: int
    rotation: int
    scale: int
    size_mm: Size
    size_px: Size


class MapRenderInput(Data):
    background_color: int
    items: List['MapRenderInputItem']
    view: 'MapRenderView'


class MapRenderOutputItemType(Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


class MapRenderOutputItem(Data):
    path: str
    tags: List['Tag']
    image: 'IImage'
    type: 'MapRenderOutputItemType'


class MapRenderOutput(Data):
    base_dir: str
    items: List['MapRenderOutputItem']
    view: 'MapRenderView'


class TemplateRenderArgs(Data):
    mro: Optional['MapRenderOutput']
    out_path: Optional[str]
    in_path: Optional[str]
    legends: Optional[dict]
    format: Optional[str]


class TemplateOutput(Data):
    content: Union[bytes, str]
    mime: str
    path: str


class ITemplateBundle(INode, Protocol):
    items: List['ITemplate']

    def find(self, subject: str = None, category: str = None, mime: str = None) -> Optional['ITemplate']: ...


class ITemplate(INode, Protocol):
    category: str
    data_model: Optional['IDataModel']
    key: str
    mime_types: List[str]
    path: str
    subject: str
    text: str

    def render(self, context: dict, args: TemplateRenderArgs = None) -> TemplateOutput: ...


# ----------------------------------------------------------------------------------------------------------------------
# styles


# noinspection PyPropertyDefinition
class IStyle(INode, Protocol):
    @property
    def values(self) -> Data: ...


# ----------------------------------------------------------------------------------------------------------------------
# metadata

# noinspection PyPropertyDefinition
class IMetaData(INode, Protocol):
    @property
    def values(self) -> Data: ...

    def extend(self, other): ...

    def set(self, key: str, value): ...

    def get(self, key: str): ...


# ----------------------------------------------------------------------------------------------------------------------
# search

class SearchFilter(Data):
    name: str
    operator: str
    shape: 'IShape'
    sub: List['SearchFilter']
    value: str


class SearchArgs(Data):
    axis: str
    bounds: Bounds
    keyword: Optional[str]
    filter: Optional['SearchFilter']
    layers: List['ILayer']
    limit: int
    params: dict
    project: 'IProject'
    resolution: float
    shapes: List['IShape']
    source_layer_names: List[str]
    tolerance: 'Measurement'


class ISearchProvider(INode, Protocol):
    data_model: Optional['IDataModel']

    supports_filter: bool = False
    supports_geometry: bool = False
    supports_keyword: bool = False

    with_filter: bool
    with_geometry: bool
    with_keyword: bool

    templates: Optional['ITemplateBundle']
    tolerance: 'Measurement'

    def run(self, args: SearchArgs, layer: 'ILayer' = None) -> List['IFeature']: ...

    def can_run(self, args: SearchArgs) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers

# noinspection PyPropertyDefinition
class IMap(INode, Protocol):
    layers: List['ILayer']

    center: Point
    coordinate_precision: int
    crs: Crs
    extent: Extent
    init_resolution: float
    resolutions: List[float]

    @property
    def bounds(self) -> 'Bounds': ...


class Legend(Data):
    cache_max_age: int
    enabled: bool
    options: dict
    path: str
    template: Optional['ITemplate']
    urls: List[Url]
    layers: List['ILayer']


class LegendRenderOutput(Data):
    html: str
    image: bytes
    image_path: str


# noinspection PyPropertyDefinition
class ILayer(INode, Protocol):
    map: 'IMap'
    metadata: 'IMetaData'

    can_render_box: bool
    can_render_xyz: bool
    can_render_svg: bool

    is_group: bool
    is_public: bool
    is_editable: bool

    supports_wms: bool
    supports_wfs: bool

    legend: 'Legend'

    image_format: str
    display: str

    layers: List['ILayer'] = []

    templates: 'ITemplateBundle'
    data_model: Optional['IDataModel']
    style: Optional['IStyle']
    search_providers: List['ISearchProvider']

    resolutions: List[float]
    extent: Extent
    opacity: float
    geometry_type: Optional[GeometryType]
    crs: Crs

    client_options: Data

    ows_name: str
    ows_feature_name: str

    edit_data_model: Optional['IDataModel']
    edit_options: Optional[Data]
    edit_style: Optional['IStyle']

    @property
    def description(self) -> str: ...

    @property
    def has_search(self) -> bool: ...

    @property
    def has_legend(self) -> bool: ...

    @property
    def has_cache(self) -> bool: ...

    @property
    def own_bounds(self) -> Optional[Bounds]: ...

    @property
    def legend_url(self) -> Url: ...

    @property
    def ancestors(self) -> List['ILayer']: ...

    def render_box(self, rv: MapRenderView, extra_params=None) -> bytes: ...

    def render_xyz(self, x: int, y: int, z: int) -> bytes: ...

    def render_svg(self, rv: 'MapRenderView', style: Optional['IStyle']) -> str: ...

    def render_svg_tags(self, rv: 'MapRenderView', style: Optional['IStyle']) -> List['Tag']: ...

    def get_legend(self, context: dict = None) -> Optional[LegendRenderOutput]: ...

    def render_legend(self, context: dict = None) -> Optional[LegendRenderOutput]: ...

    def get_features(self, bounds: Bounds, limit: int = 0) -> List['IFeature']: ...

    def enabled_for_ows(self, service: 'IOwsService') -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# OWS

class OwsProtocol(Enum):
    WMS = 'WMS'
    WMTS = 'WMTS'
    WCS = 'WCS'
    WFS = 'WFS'
    CSW = 'CSW'


class OwsVerb(Enum):
    CreateStoredQuery = 'CreateStoredQuery'
    DescribeFeatureType = 'DescribeFeatureType'
    DescribeLayer = 'DescribeLayer'
    DescribeStoredQueries = 'DescribeStoredQueries'
    DropStoredQuery = 'DropStoredQuery'
    GetCapabilities = 'GetCapabilities'
    GetFeature = 'GetFeature'
    GetFeatureInfo = 'GetFeatureInfo'
    GetFeatureWithLock = 'GetFeatureWithLock'
    GetLegendGraphic = 'GetLegendGraphic'
    GetMap = 'GetMap'
    GetPrint = 'GetPrint'
    GetPropertyValue = 'GetPropertyValue'
    GetTile = 'GetTile'
    ListStoredQueries = 'ListStoredQueries'
    LockFeature = 'LockFeature'
    Transaction = 'Transaction'


class OwsOperation(Data):
    formats: List[str]
    get_url: Url
    params: Dict
    post_url: Url
    verb: OwsVerb


class IOwsService(INode, Protocol):
    metadata: 'IMetaData'
    name: str
    protocol: OwsProtocol
    supported_crs: List[Crs]
    supported_versions: List[str]
    templates: 'ITemplateBundle'
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    def handle_request(self, req: 'IWebRequest') -> ContentResponse: ...

    def error_response(self, err: Exception) -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    metadata: 'IMetaData'
    protocol: OwsProtocol
    supported_crs: List[Crs]
    url: Url
    version: str


# ----------------------------------------------------------------------------------------------------------------------


# CLI

class CliParams(Data):
    """CLI params"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# projects and application


class IMonitor(INode, Protocol):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_path(self, path: str): ...

    def start(self): ...


class IClient(INode, Protocol):
    pass


class IApi(INode, Protocol):
    def actions_for(self, user: 'IGrantee', parent: 'IApi' = None) -> Dict[str, INode]: ...


class IProject(INode, Protocol):
    api: 'IApi'
    assets_root: Optional['DocumentRoot']
    client: 'IClient'
    locale_uids: List[str]
    map: 'IMap'
    metadata: 'IMetaData'
    search_providers: List['ISearchProvider']
    templates: 'ITemplateBundle'


class IApplication(INode, Protocol):
    api: 'IApi'
    auth: 'IAuthManager'
    client: 'IClient'
    locale_uids: List[str]
    metadata: 'IMetaData'
    monitor: 'IMonitor'
    mpx_url: str
    web_sites: List['IWebSite']
    qgis_version: str
    version: str

    def developer_option(self, name: str): ...

    def find_action(self, user: 'IGrantee', ext_type: str, project_uid: str = None) -> Optional[INode]: ...

    def require_helper(self, ext_type: str): ...

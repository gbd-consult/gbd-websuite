from gws.types import Any, Dict, Enum, List, Literal, Optional, Protocol, Tuple, Union
from .data import Data

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


##


class Bounds(Data):
    crs: Crs
    extent: Extent


# ----------------------------------------------------------------------------------------------------------------------
# basic objects

Klass = Union[type, str]


class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class Access(Config):
    """Access rights definition for authorization roles"""

    type: Literal['allow', 'deny']  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithAccess(Config):
    access: Optional[List[Access]]  #: access rights


class Props(Data):
    """Properties base type"""
    pass


# noinspection PyPropertyDefinition
class IObject(Protocol):
    access: Optional['Access']
    class_name: str
    ext_type: str
    title: str
    uid: str
    configured: bool

    @property
    def props(self) -> 'Props': ...

    def props_for(self, user: 'IUser') -> Optional['Props']: ...

    def configure(self): ...


# noinspection PyPropertyDefinition
class INode(IObject, Protocol):
    root: 'IRootObject'
    parent: 'INode'

    def var(self, key: str, default=None, with_parent=False): ...

    def create_child(self, klass: Klass, cfg: Optional[Any]) -> 'IObject': ...


class ExtDescriptor(Data):
    name: str
    ext_type: str
    module_name: str
    module_path: str
    ident: str
    class_ptr: type


class ExtCommandDescriptor(Data):
    params: 'Params'
    action_type: str
    function_name: str
    class_name: str


class ISpecRuntime(Protocol):
    def check_command(self, cmd: str, method: str, params, strict=True) -> Optional[ExtCommandDescriptor]: ...

    def read_value(self, value, type_name: str, path='', strict=True) -> Any: ...

    def get_ext_descriptor(self, class_name: str) -> Optional[ExtDescriptor]: ...

    def objects(self, pattern: str) -> List[Data]: ...

    def client_vendor_bundle_path(self) -> str: ...

    def client_bundle_paths(self) -> List[str]: ...


class IRootObject(IObject, Protocol):
    application: 'IApplication'
    specs: 'ISpecRuntime'

    def find_all(self, klass: Klass = None, uid: str = None, ext_type: str = None) -> List['INode']: ...

    def find(self, klass: Klass = None, uid: str = None, ext_type: str = None) -> Optional['INode']: ...


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


# noinspection PyPropertyDefinition
class IWebRequest(Protocol):
    site: 'IWebSite'

    @property
    def is_secure(self) -> bool: ...

    @property
    def user(self) -> 'IUser': ...

    def acquire(self, klass: str, uid: str) -> Optional['IObject']: ...

    def require(self, klass: str, uid: Optional[str]) -> 'IObject': ...

    def require_layer(self, uid: Optional[str]) -> 'ILayer': ...

    def require_project(self, uid: Optional[str]) -> 'IProject': ...

    def has_param(self, key: str) -> bool: ...

    def param(self, key: str, default: str = None) -> str: ...

    def cookie(self, key: str, default: str = None) -> str: ...

    def header(self, key: str, default: str = None) -> str: ...

    def env(self, key: str, default: str = None) -> str: ...


class IWebResponse(Protocol):
    def set_cookie(self, key: str, **kwargs): ...

    def delete_cookie(self, key: str, **kwargs): ...

    def add_header(self, key: str, value): ...


# ----------------------------------------------------------------------------------------------------------------------
# web sites


class IDocumentRoot(INode, Protocol):
    dir: DirPath
    allow_mime: Optional[List[str]]
    deny_mime: Optional[List[str]]


class IWebSite(INode, Protocol):
    assets_root: Optional['IDocumentRoot']

    def url_for(self, req: 'IWebRequest', url: Url) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization

# noinspection PyPropertyDefinition
class IAuthManager(INode, Protocol):
    guest_user: 'IUser'

    def get_user(self, user_fid: str) -> Optional['IUser']: ...

    def get_role(self, name: str) -> 'IRole': ...


class IRole(Protocol):
    name: str
    uid: str

    def can_use(self, obj: IObject, parent: IObject = None) -> bool: ...


# noinspection PyPropertyDefinition
class IUser(Protocol):
    name: str
    uid: str
    attributes: Dict[str, Any]

    @property
    def props(self) -> 'Props': ...

    @property
    def display_name(self) -> str: ...

    @property
    def is_guest(self) -> bool: return False

    @property
    def fid(self) -> str: ...

    def can_use(self, obj: IObject, parent: IObject = None) -> bool: ...


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
    def bounds(self) -> 'Bounds': ...

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
    def type(self) -> 'GeometryType': ...

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

    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape': ...

    def transformed_to(self, to_crs, **kwargs) -> 'IShape': ...


class IFeature(IObject, Protocol):
    data_model: Optional['IDataModel']
    layer: Optional['ILayer']
    shape: Optional['IShape']
    style: Optional['IStyle']
    templates: Optional['ITemplateBundle']

    attributes: List['Attribute']
    category: str

    def apply_data_model(self, model: 'IDataModel' = None) -> 'IFeature': ...

    def apply_templates(self, templates: 'ITemplateBundle' = None, extra_context: dict = None, keys: List[str] = None) -> 'IFeature': ...

    def attr(self, name: str): ...

    def to_geojson(self) -> dict: ...

    def to_svg(self, rv: 'MapRenderView', style: 'IStyle' = None) -> str: ...

    def to_svg_tags(self, rv: 'MapRenderView', style: 'IStyle' = None) -> List['Tag']: ...

    def transform_to(self, crs: Crs) -> 'IFeature': ...


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


class IImage(Protocol):
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
    bounds: 'Bounds'
    center: 'Point'
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
    content: str
    mime: str
    path: str


class ITemplateBundle(INode, Protocol):
    def all(self) -> List['ITemplate']: ...

    def find(self, subject: str = None, category: str = None, mime: str = None) -> Optional['ITemplate']: ...


class ITemplate(INode, Protocol):
    category: str
    key: str
    mime_types: List[str]
    path: str
    subject: str
    text: str

    data_model: Optional['IDataModel']

    def render(self, context: dict, args: TemplateRenderArgs = None) -> TemplateOutput: ...


# ----------------------------------------------------------------------------------------------------------------------
# styles


class StyleData(Data):
    pass  # the actual object is in lib.style


class IStyle(INode, Protocol):
    data: StyleData


# ----------------------------------------------------------------------------------------------------------------------
# metadata

class MetaData(Data):
    pass  # the actual object is in lib.meta


class IMeta(INode, Protocol):
    data: MetaData

    def extend(self, meta): ...


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


class ISearchProvider(IObject, Protocol):
    templates: 'ITemplateBundle'
    data_model: Optional['IDataModel']

    def run(self, args: SearchArgs, layer: 'ILayer' = None) -> List['IFeature']: ...

    def can_run(self, args: SearchArgs) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers

class IMap(INode):
    layers: List['ILayer']

    center: Point
    bounds: Bounds
    coordinate_precision: int
    crs: Crs
    extent: Extent
    init_resolution: float
    resolutions: List[float]


class Legend(Data):
    enabled: bool
    path: str
    url: str
    template: Optional['ITemplate']


# noinspection PyPropertyDefinition
class ILayer(INode, Protocol):
    map: 'IMap'
    meta: 'IMeta'

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

    def render_legend_to_path(self, context: dict = None) -> Optional[str]: ...

    def render_legend_to_image(self, context: dict = None) -> Optional[bytes]: ...

    def render_legend_to_html(self, context: dict = None) -> Optional[str]: ...

    def get_features(self, bounds: Bounds, limit: int = 0) -> List['IFeature']: ...

    def enabled_for_ows(self, service: 'IOwsService') -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# OWS

class IOwsService(INode, Protocol):
    meta: 'IMeta'
    name: str
    service_type: str
    supported_crs: List[Crs]
    supported_versions: List[str]
    templates: 'ITemplateBundle'
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    def handle_request(self, req: 'IWebRequest') -> ContentResponse: ...

    def error_response(self, err: Exception) -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    meta: 'IMeta'
    service_type: str
    supported_crs: List[Crs]
    url: Url
    version: str


# ----------------------------------------------------------------------------------------------------------------------
# CLI

class CliParams(Data):
    """CLI params"""

    verbose: bool = False
    manifest: str = ''


# ----------------------------------------------------------------------------------------------------------------------
# projects and application


class IMonitor(INode):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_path(self, path: str): ...

    def start(self): ...


class IProject(INode, Protocol):
    assets_root: Optional['IDocumentRoot']
    locale_uids: List[str]
    search_providers: List['ISearchProvider']
    map: 'IMap'
    meta: 'IMeta'
    templates: 'ITemplateBundle'

    def find_action(self, action_type: str) -> Optional[IObject]: ...


class IApplication(INode, Protocol):
    auth: 'IAuthManager'
    monitor: 'IMonitor'
    meta: 'IMeta'
    localeUids: List[str]

    web_sites: List['IWebSite']

    def developer_option(self, name: str) -> Any: ...

    def find_action(self, action_type: str, project_uid: str = None) -> Optional[IObject]: ...

    def helper(self, key: str) -> INode: ...

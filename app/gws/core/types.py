from .data import Data
from gws.types import Any, Callable, Dict, Enum, List, Literal, Optional, Protocol, Set, Tuple, Union

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

#: A Point with a unit
MPoint = Tuple[float, float, str]

#: A Size with a unit
MSize = Tuple[float, float, str]

#: An XML generator tag
Tag = tuple

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

#: ISO date like "2019-01-30"
Date = str

#: ISO date/time like "2019-01-30 01:02:03"
DateTime = str

#: Http or https URL
Url = str


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

    def props_for(self, user: 'IGrantee') -> Props: ...

    def access_for(self, user: 'IGrantee') -> Optional[bool]: ...

    def is_a(self, klass: Klass) -> bool: ...


class IGrantee(Protocol):
    roles: Set[str]

    def can_use(self, obj: 'IObject', context: 'IObject' = None) -> bool: ...

    def require(self, klass: Optional[Klass], uid: Optional[str]): ...

    def acquire(self, klass: Optional[Klass], uid: Optional[str]): ...


class INode(IObject, Protocol):
    children: List['INode']
    config: Config
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
    def environ(self) -> dict: ...

    @property
    def data(self) -> Optional[bytes]: ...

    @property
    def text(self) -> Optional[str]: ...

    @property
    def is_secure(self) -> bool: ...

    @property
    def method(self) -> str: ...

    @property
    def user(self) -> 'IUser': ...

    def acquire(self, klass: str, uid: Optional[str]): ...

    def cookie(self, key: str, default: str = '') -> str: ...

    def env(self, key: str, default: str = '') -> str: ...

    def has_param(self, key: str) -> bool: ...

    def header(self, key: str, default: str = '') -> str: ...

    def param(self, key: str, default: str = '') -> str: ...

    def require(self, klass: Optional[Klass], uid: Optional[str]): ...

    def require_layer(self, uid: Optional[str]) -> 'ILayer': ...

    def require_project(self, uid: Optional[str]) -> 'IProject': ...

    def url_for(self, path: str, **params) -> Url: ...


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

    def url_for(self, req: 'IWebRequest', path: str, **params) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization

class IAuthManager(INode, Protocol):
    guest_user: 'IUser'
    system_user: 'IUser'
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


class AttributeEditor(Data):
    type: str
    accept: Optional[str]
    items: Optional[Any]
    max: Optional[float]
    min: Optional[float]
    multiple: Optional[bool]
    pattern: Optional[Regex]


class IDataModel(INode, Protocol):
    def apply(self, attributes: List['Attribute']) -> List['Attribute']: ...

    def apply_to_dict(self, attr_values: dict) -> List['Attribute']: ...

    def xml_schema_dict(self, name_for_geometry) -> dict: ...


# ----------------------------------------------------------------------------------------------------------------------
# CRS

#: Axis orientation
Axis = int

AXIS_XY = 1
AXIS_YX = 2

#: CRS code like "EPSG:3857" or a srid like 3857
CrsId = Union[str, int]


class CrsFormat(Enum):
    SRID = 'SRID'
    EPSG = 'EPSG'
    URL = 'URL'
    URI = 'URI'
    URNX = 'URNX'
    URN = 'URN'


class Bounds(Data):
    crs: 'ICrs'
    extent: Extent


# noinspection PyPropertyDefinition
class ICrs(IObject, Protocol):
    srid: str
    proj4text: int
    units: str
    is_geographic: bool
    is_projected: bool

    epsg: str
    urn: str
    urnx: str
    url: str
    uri: str

    def transform_extent(self, extent: Extent, target: 'ICrs') -> Extent: ...

    def transform_geometry(self, geom: dict, target: 'ICrs') -> dict: ...

    def to_string(self, fmt: CrsFormat) -> str: ...

    def to_geojson(self) -> dict: ...


# ----------------------------------------------------------------------------------------------------------------------
# Geodata sources

class TileMatrix(Data):
    uid: str
    scale: float
    x: float
    y: float
    width: float
    height: float
    tile_width: float
    tile_height: float
    extent: Extent


class TileMatrixSet(Data):
    uid: str
    crs: 'ICrs'
    matrices: List[TileMatrix]


class SourceStyle(Data):
    is_default: bool
    legend_url: Url
    metadata: 'IMetadata'
    name: str


class SourceLayer(Data):
    a_level: int
    a_path: str
    a_uid: str

    data_source: dict

    supported_bounds: List[Bounds]

    is_expanded: bool
    is_group: bool
    is_image: bool
    is_queryable: bool
    is_visible: bool

    layers: List['SourceLayer']

    metadata: 'IMetadata'
    name: str
    title: str

    legend_url: Url
    opacity: int
    scale_range: List[float]

    styles: List[SourceStyle]
    default_style: Optional[SourceStyle]

    tile_matrix_ids: List[str]
    tile_matrix_sets: List[TileMatrixSet]
    image_format: str
    resource_urls: dict


# ----------------------------------------------------------------------------------------------------------------------
# XML

class XmlElement:
    name: str
    children: List['XmlElement']
    attributes: Dict[str, Any]
    text: str
    tail: str


# ----------------------------------------------------------------------------------------------------------------------
# shapes and features

# noinspection PyPropertyDefinition
class IShape(IObject, Protocol):
    crs: 'ICrs'

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

    def to_geojson(self) -> dict: ...

    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape': ...

    def transformed_to(self, crs: 'ICrs') -> 'IShape': ...


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

    def apply_templates(self, templates: 'ITemplateBundle' = None, extra_context: dict = None, subjects: List[str] = None) -> 'IFeature': ...

    def attr(self, name: str): ...

    def to_geojson(self) -> dict: ...

    def to_svg_element(self, view: 'MapView', style: 'IStyle' = None) -> Optional[XmlElement]: ...

    def to_svg_fragment(self, view: 'MapView', style: 'IStyle' = None) -> List[XmlElement]: ...

    def transform_to(self, crs: 'ICrs') -> 'IFeature': ...

    def connect_to(self, layer: 'ILayer') -> 'IFeature': ...


# ----------------------------------------------------------------------------------------------------------------------
# database


class SqlTable(Data):
    name: str
    key_column: str
    search_column: str
    geometry_column: str
    geometry_type: GeometryType
    geometry_crs: 'ICrs'


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
    crs: 'ICrs'
    srid: int
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

# noinspection PyPropertyDefinition
class IImage(IObject, Protocol):
    @property
    def size(self) -> Size: ...

    def add_box(self, color=None) -> 'IImage': ...

    def add_text(self, text: str, x=0, y=0, color=None) -> 'IImage': ...

    def compose(self, other: 'IImage', opacity=1) -> 'IImage': ...

    def crop(self, box) -> 'IImage': ...

    def paste(self, other: 'IImage', where=None) -> 'IImage': ...

    def resize(self, size: Size, **kwargs) -> 'IImage': ...

    def rotate(self, angle: int, **kwargs) -> 'IImage': ...

    def to_bytes(self, mime: str = None) -> bytes: ...

    def to_path(self, path: str, mime: str = None) -> str: ...


class MapView(Data):
    bounds: Bounds
    center: Point
    rotation: int
    scale: int
    size_mm: Size
    size_px: Size
    dpi: int


class MapRenderInputPlane(Data):
    type: Literal['features', 'image', 'image_layer', 'svg_layer', 'svg_soup']
    features: List['IFeature']
    image: 'IImage'
    layer: 'ILayer'
    opacity: float
    print_as_vector: bool
    soup_points: List[Point]
    soup_tags: List[Any]
    style: 'IStyle'
    sub_layers: List[str]


class MapRenderInput(Data):
    background_color: int
    bbox: Extent
    center: Point
    crs: 'ICrs'
    dpi: int
    out_size: MSize
    out_path: str
    planes: List['MapRenderInputPlane']
    rotation: int
    scale: int


class MapRenderOutputPlane(Data):
    type: Literal['image', 'path', 'svg']
    path: str
    elements: List[XmlElement]
    image: 'IImage'


class MapRenderOutput(Data):
    path: str
    planes: List['MapRenderOutputPlane']
    view: MapView


class TemplateRenderInputMap(Data):
    background_color: int
    bbox: Extent
    center: Point
    planes: List['MapRenderInputPlane']
    rotation: int
    scale: int
    visible_layers: Optional[List['ILayer']]


class TemplateRenderInput(Data):
    context: Optional[dict]
    crs: 'ICrs'
    dpi: Optional[int]
    locale_uid: Optional[str]
    maps: Optional[List[TemplateRenderInputMap]]
    out_mime: Optional[str]
    out_path: Optional[str]
    user: Optional['IUser']


class TemplateQualityLevel(Data):
    name: str
    dpi: int


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
    image: 'IImage'
    image_path: str
    size: Size


class ITemplate(INode, Protocol):
    category: str
    data_model: Optional['IDataModel']
    mimes: List[str]
    name: str
    path: str
    subject: str
    text: str
    quality_levels: List[TemplateQualityLevel]

    map_size: Optional[MSize]
    page_size: Optional[MSize]

    def render(self, tri: TemplateRenderInput, notify: Callable = None) -> ContentResponse: ...


class ITemplateBundle(INode, Protocol):
    items: List['ITemplate']

    def find(self, subject: str = None, category: str = None, name: str = None, mime: str = None) -> Optional['ITemplate']: ...


class IPrinter(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# styles

class StyleValues(Data):
    fill: Color

    stroke: Color
    stroke_dasharray: List[int]
    stroke_dashoffset: int
    stroke_linecap: Literal['butt', 'round', 'square']
    stroke_linejoin: Literal['bevel', 'round', 'miter']
    stroke_miterlimit: int
    stroke_width: int

    marker: Literal['circle', 'square', 'arrow', 'cross']
    marker_fill: Color
    marker_size: int
    marker_stroke: Color
    marker_stroke_dasharray: List[int]
    marker_stroke_dashoffset: int
    marker_stroke_linecap: Literal['butt', 'round', 'square']
    marker_stroke_linejoin: Literal['bevel', 'round', 'miter']
    marker_stroke_miterlimit: int
    marker_stroke_width: int

    with_geometry: Literal['all', 'none']
    with_label: Literal['all', 'none']

    label_align: Literal['left', 'right', 'center']
    label_background: Color
    label_fill: Color
    label_font_family: str
    label_font_size: int
    label_font_style: Literal['normal', 'italic']
    label_font_weight: Literal['normal', 'bold']
    label_line_height: int
    label_max_scale: int
    label_min_scale: int
    label_offset_x: int
    label_offset_y: int
    label_padding: List[int]
    label_placement: Literal['start', 'end', 'middle']
    label_stroke: Color
    label_stroke_dasharray: List[int]
    label_stroke_dashoffset: int
    label_stroke_linecap: Literal['butt', 'round', 'square']
    label_stroke_linejoin: Literal['bevel', 'round', 'miter']
    label_stroke_miterlimit: int
    label_stroke_width: int

    point_size: int
    icon: str

    offset_x: int
    offset_y: int


class IStyle(IObject, Protocol):
    name: str
    selector: str
    text: str
    values: StyleValues


# ----------------------------------------------------------------------------------------------------------------------
# locale

class Locale(Data):
    id: str
    dateFormatLong: str
    dateFormatMedium: str
    dateFormatShort: str
    dateUnits: str  #: date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'
    dayNamesLong: List[str]
    dayNamesShort: List[str]
    dayNamesNarrow: List[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthNamesLong: List[str]
    monthNamesShort: List[str]
    monthNamesNarrow: List[str]
    numberDecimal: str
    numberGroup: str


# ----------------------------------------------------------------------------------------------------------------------
# metadata


class MetadataLink(Data):
    """Link metadata"""

    scheme: str
    url: Url
    formatName: str
    formatVersion: str
    function: str
    type: str


class MetadataValues(Data):
    abstract: str
    accessConstraints: str
    attribution: str

    authorityIdentifier: str
    authorityName: str
    authorityUrl: str

    catalogCitationUid: str
    catalogUid: str

    contactAddress: str
    contactArea: str
    contactCity: str
    contactCountry: str
    contactEmail: str
    contactFax: str
    contactOrganization: str
    contactPerson: str
    contactPhone: str
    contactPosition: str
    contactProviderName: str
    contactProviderSite: str
    contactRole: str
    contactUrl: str
    contactZip: str

    dateBegin: str
    dateCreated: str
    dateEnd: str
    dateUpdated: str

    fees: str
    image: str

    inspireKeywords: List[str]
    inspireMandatoryKeyword: str
    inspireDegreeOfConformity: str
    inspireResourceType: str
    inspireSpatialDataServiceType: str
    inspireSpatialScope: str
    inspireSpatialScopeName: str
    inspireTheme: str
    inspireThemeName: str
    inspireThemeNameEn: str

    isoMaintenanceFrequencyCode: str
    isoQualityConformanceExplanation: str
    isoQualityConformanceQualityPass: bool
    isoQualityConformanceSpecificationDate: str
    isoQualityConformanceSpecificationTitle: str
    isoQualityLineageSource: str
    isoQualityLineageSourceScale: int
    isoQualityLineageStatement: str
    isoRestrictionCode: str
    isoScope: str
    isoScopeName: str
    isoSpatialRepresentationType: str
    isoTopicCategory: str
    isoSpatialResolution: str

    keywords: List[str]
    language3: str
    language: str
    languageName: str
    license: str
    name: str
    title: str

    metaLinks: List[MetadataLink]
    extraLinks: List[MetadataLink]

    crs: 'ICrs'
    extent4326: Extent
    boundingPolygonElement: XmlElement


class IMetadata(IObject, Protocol):
    values: MetadataValues

    def extend(self, *others) -> 'IMetadata': ...

    def set(self, key: str, value) -> 'IMetadata': ...

    def get(self, key: str, default=None): ...


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
    crs: 'ICrs'
    extent: Extent
    init_resolution: float
    resolutions: List[float]

    @property
    def bounds(self) -> 'Bounds': ...


# noinspection PyPropertyDefinition
class ILayer(INode, Protocol):
    map: 'IMap'
    metadata: 'IMetadata'

    can_render_box: bool
    can_render_xyz: bool
    can_render_svg: bool

    is_group: bool
    is_editable: bool

    supports_raster_ows: bool
    supports_vector_ows: bool

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
    crs: 'ICrs'

    client_options: Data

    ows_enabled: bool

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

    def render_box(self, view: MapView, extra_params=None) -> bytes: ...

    def render_xyz(self, x: int, y: int, z: int) -> bytes: ...

    def render_svg_element(self, view: 'MapView', style: Optional['IStyle']) -> Optional[XmlElement]: ...

    def render_svg_fragment(self, view: 'MapView', style: Optional['IStyle']) -> List[XmlElement]: ...

    def render_legend_with_cache(self, context: dict = None) -> Optional[LegendRenderOutput]: ...

    def render_legend(self, context: dict = None) -> Optional[LegendRenderOutput]: ...

    def get_features(self, bounds: Bounds, limit: int = 0) -> List['IFeature']: ...


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
    DescribeCoverage = 'DescribeCoverage'
    DescribeFeatureType = 'DescribeFeatureType'
    DescribeLayer = 'DescribeLayer'
    DescribeRecord = 'DescribeRecord'
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
    GetRecordById = 'GetRecordById'
    GetRecords = 'GetRecords'
    GetTile = 'GetTile'
    ListStoredQueries = 'ListStoredQueries'
    LockFeature = 'LockFeature'
    Transaction = 'Transaction'


class OwsOperation(Data):
    formats: List[str]
    get_url: Url
    params: Dict[str, List[str]]
    post_url: Url
    verb: OwsVerb


class IOwsService(INode, Protocol):
    metadata: 'IMetadata'
    name: str
    protocol: OwsProtocol
    supported_bounds: List[Bounds]
    supported_versions: List[str]
    templates: 'ITemplateBundle'
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    def handle_request(self, req: 'IWebRequest') -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    metadata: 'IMetadata'
    operations: List['OwsOperation']
    protocol: OwsProtocol
    source_layers: List['SourceLayer']
    url: Url
    version: str
    force_crs: 'ICrs'


class IOwsClient(INode, Protocol):
    provider: 'IOwsProvider'
    source_layers: List['SourceLayer']


# ----------------------------------------------------------------------------------------------------------------------


# CLI

class CliParams(Data):
    """CLI params"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# actions and apis


class IAction(INode, Protocol):
    pass


class IApi(INode, Protocol):
    def actions_for(self, user: 'IGrantee', parent: 'IApi' = None) -> Dict[str, INode]: ...


# ----------------------------------------------------------------------------------------------------------------------
# projects


class IClient(INode, Protocol):
    pass


class IProject(INode, Protocol):
    api: 'IApi'
    assets_root: Optional['DocumentRoot']
    client: 'IClient'
    locale_uids: List[str]
    map: 'IMap'
    metadata: 'IMetadata'
    search_providers: List['ISearchProvider']
    templates: 'ITemplateBundle'


# ----------------------------------------------------------------------------------------------------------------------
# application

class IMonitor(INode, Protocol):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_path(self, path: str): ...

    def start(self): ...


class IApplication(INode, Protocol):
    api: 'IApi'
    auth: 'IAuthManager'
    client: 'IClient'
    locale_uids: List[str]
    metadata: 'IMetadata'
    monitor: 'IMonitor'
    mpx_url: str
    web_sites: List['IWebSite']
    qgis_version: str
    version: str

    def developer_option(self, name: str): ...

    def find_action(self, user: 'IGrantee', ext_type: str, project_uid: str = None) -> Optional[INode]: ...

    def require_helper(self, ext_type: str): ...

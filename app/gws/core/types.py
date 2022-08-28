from .data import Data
from gws.types import Any, Callable, Dict, Enum, List, Literal, Optional, Protocol, Set, Tuple, Union

# ----------------------------------------------------------------------------------------------------------------------
# custom types, used everywhere


"""type: An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]."""
Extent = Tuple[float, float, float, float]

"""type: Point coordinates [x, y]."""
Point = Tuple[float, float]

"""type: Size [width, height]."""
Size = Tuple[float, float]

"""type: A value with a unit."""
Measurement = Tuple[float, str]

"""type: A Point with a unit."""
MPoint = Tuple[float, float, str]

"""type: A Size with a unit."""
MSize = Tuple[float, float, str]

"""type: An XML generator tag."""
Tag = tuple

"""type: Valid readable file path on the server."""
FilePath = str

"""type: Valid readable directory path on the server."""
DirPath = str

"""type: String like "1w 2d 3h 4m 5s" or a number of seconds."""
Duration = str

"""type: CSS color name."""
Color = str

"""type: Regular expression, as used in Python."""
Regex = str

"""type: String with {attribute} placeholders."""
FormatStr = str

"""type: ISO date like "2019-01-30"."""
Date = str

"""type: ISO date/time like "2019-01-30 01:02:03"."""
DateTime = str

"""type: Http or https URL."""
Url = str


# ----------------------------------------------------------------------------------------------------------------------
# application manifest


class ApplicationManifestPlugin(Data):
    path: FilePath
    name: str = ''


class ApplicationManifest(Data):
    excludePlugins: Optional[List[str]]
    plugins: Optional[List[ApplicationManifestPlugin]]
    locales: List[str]

    withFallbackConfig: bool = False
    withStrictConfig: bool = False


# ----------------------------------------------------------------------------------------------------------------------
# basic objects

ClassRef = Union[type, str]


class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


"""type: Access Control List."""
Access = List[Tuple[int, str]]

"""type: A string of comma-separated pairs 'allow <role>' or 'deny <role>'."""
ACL = str


class ConfigWithAccess(Config):
    access: Optional[ACL]  #: access rights


class Props(Data):
    """Properties base type"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# foundation interfaces

class IObject(Protocol):
    extName: str
    extType: str
    access: Access
    uid: str

    def props(self, user: 'IGrantee') -> Props: ...


class IGrantee(Protocol):
    roles: Set[str]

    def can_use(self, obj: Any, context=None) -> bool: ...

    def access_to(self, obj: Any) -> Optional[int]: ...


class INode(IObject, Protocol):
    config: Config
    root: 'IRoot'
    parent: 'INode'
    title: str
    uid: str

    def activate(self): ...

    def configure(self): ...

    def create_child(self, classref: ClassRef, config=None, optional=False, required=False): ...

    def create_children(self, classref: ClassRef, configs: Any, required=False): ...

    def post_configure(self): ...

    def pre_configure(self): ...

    def var(self, key: str, default=None): ...


class IRoot(Protocol):
    app: 'IApplication'
    specs: 'ISpecRuntime'
    configErrors: List[str]

    def post_initialize(self): ...

    def activate(self): ...

    def find_all(self, classref: ClassRef) -> List: ...

    def find(self, classref: ClassRef, uid: str): ...

    def create(self, classref: ClassRef, config=None, optional=False, required=False): ...

    def create_child(self, parent: 'INode', classref: ClassRef, config=None, optional=False, required=False): ...

    def create_shared(self, classref: ClassRef, config=None): ...

    def create_application(self, config=None) -> 'IApplication': ...


# ----------------------------------------------------------------------------------------------------------------------
# spec runtime


class ExtObjectDescriptor(Data):
    extName: str
    classPtr: type
    ident: str
    modName: str
    modPath: str


class ExtCommandDescriptor(Data):
    extName: str
    methodName: str
    methodPtr: Callable
    request: 'Request'
    tArg: str
    tOwner: str


class ISpecRuntime(Protocol):
    version: str
    manifest: ApplicationManifest

    def read(self, value: Any, type_name: str, path: str = '', strict_mode=True, verbose_errors=True, accept_extra_props=False) -> Any: ...

    def object_descriptor(self, type_name: str) -> Optional[ExtObjectDescriptor]: ...

    def command_descriptor(self, command_category: str, command_name: str) -> Optional[ExtCommandDescriptor]: ...

    def cli_docs(self, lang: str = 'en') -> dict: ...

    def bundle_paths(self, category: str) -> List[str]: ...

    def parse_classref(self, classref: ClassRef) -> Tuple[Optional[type], str, str]: ...


# ----------------------------------------------------------------------------------------------------------------------
# requests and responses

class Request(Data):
    """Web request"""

    projectUid: Optional[str]  #: project uid
    localeUid: Optional[str]  #: locale for this request


class EmptyRequest(Data):
    """Empty web request"""
    pass


class ResponseError(Data):
    """Web response error"""
    status: int
    info: str


class Response(Data):
    """Web response"""
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


class IWebRequester(Protocol):
    site: 'IWebSite'
    environ: dict
    isSecure: bool
    isApi: bool
    isGet: bool
    isPost: bool
    method: str
    user: 'IUser'

    def cookie(self, key: str, default: str = '') -> str: ...

    def data(self) -> Optional[bytes]: ...

    def env(self, key: str, default: str = '') -> str: ...

    def has_param(self, key: str) -> bool: ...

    def header(self, key: str, default: str = '') -> str: ...

    def param(self, key: str, default: str = '') -> str: ...

    def text(self) -> Optional[str]: ...

    def url_for(self, path: str, **params) -> Url: ...

    def find(self, klass, uid: str): ...

    def require(self, classref: ClassRef, uid: str): ...

    def require_project(self, uid: str) -> 'IProject': ...

    def require_layer(self, uid: str) -> 'ILayer': ...

    def acquire(self, classref: ClassRef, uid: str): ...


class IWebResponder(Protocol):
    status_code: int

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
    assetsRoot: Optional['DocumentRoot']
    errorPage: Optional['ITemplate']

    def url_for(self, req: 'IWebRequester', path: str, **params) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization

class IAuthManager(INode, Protocol):
    guestUser: 'IUser'
    systemUser: 'IUser'
    providers: List['IAuthProvider']
    methods: List['IAuthMethod']

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def get_user(self, user_uid: str) -> Optional['IUser']: ...

    def get_provider(self, uid: str = None, ext_type: str = None) -> Optional['IAuthProvider']: ...

    def get_method(self, uid: str = None, ext_type: str = None) -> Optional['IAuthMethod']: ...

    def login(self, credentials: Data, req: IWebRequester): ...

    def logout(self, req: IWebRequester): ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...

    def session_find(self, uid: str) -> Optional['IAuthSession']: ...

    def session_create(self, typ: str, method: 'IAuthMethod', user: 'IUser') -> 'IAuthSession': ...

    def session_save(self, sess: 'IAuthSession'): ...

    def session_delete(self, sess: 'IAuthSession'): ...

    def session_delete_all(self): ...


class IAuthMethod(INode, Protocol):
    secure: bool

    def open_session(self, auth: 'IAuthManager', req: 'IWebRequester') -> Optional['IAuthSession']: ...

    def close_session(self, auth: IAuthManager, sess: 'IAuthSession', req: IWebRequester, res: IWebResponder): ...

    def login(self, auth: IAuthManager, credentials: Data, req: IWebRequester) -> Optional['IAuthSession']: ...

    def logout(self, auth: IAuthManager, sess: 'IAuthSession', req: IWebRequester) -> 'IAuthSession': ...


class IAuthProvider(INode, Protocol):
    allowedMethods: List[str]

    def get_user(self, local_uid: str) -> Optional['IUser']: ...

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...


class IAuthSession(IObject, Protocol):
    changed: bool
    saved: bool
    data: dict
    method: Optional['IAuthMethod']
    typ: str
    uid: str
    user: 'IUser'

    def get(self, key: str, default=None): ...

    def set(self, key: str, val: Any): ...


class IUser(IObject, IGrantee, Protocol):
    attributes: Dict[str, Any]
    displayName: str
    isGuest: bool
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


class IModel(INode, Protocol):
    def apply(self, attributes: List['Attribute']) -> List['Attribute']: ...

    def apply_to_dict(self, attr_values: dict) -> List['Attribute']: ...

    def xml_schema_dict(self, name_for_geometry) -> dict: ...


# ----------------------------------------------------------------------------------------------------------------------
# CRS

"""type: Axis orientation."""
Axis = int

AXIS_XY = 1
AXIS_YX = 2

"""type: CRS code like "EPSG:3857" or a srid like 3857."""
CRS = Union[str, int]


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
    data_model: Optional['IModel']
    elements: dict
    layer: Optional['ILayer']
    shape: Optional['IShape']
    style: Optional['IStyle']
    templates: Optional['ITemplateCollection']
    uid: str

    @property
    def template_context(self) -> dict: ...

    def apply_data_model(self, model: 'IModel' = None) -> 'IFeature': ...

    def apply_templates(self, templates: 'ITemplateCollection' = None, extra_context: dict = None, subjects: List[str] = None) -> 'IFeature': ...

    def attr(self, name: str): ...

    def to_geojson(self) -> dict: ...

    def to_svg_element(self, view: 'MapView', style: 'IStyle' = None) -> Optional[XmlElement]: ...

    def to_svg_fragment(self, view: 'MapView', style: 'IStyle' = None) -> List[XmlElement]: ...

    def transform_to(self, crs: 'ICrs') -> 'IFeature': ...

    def connect_to(self, layer: 'ILayer') -> 'IFeature': ...


# ----------------------------------------------------------------------------------------------------------------------
# database

class Sql:
    def __init__(self, text, *args, **kwargs):
        self.text = text
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return repr(vars(self))


class SqlColumn(Data):
    name: str
    type: AttributeType
    gtype: GeometryType
    native_type: str
    crs: 'ICrs'
    srid: int
    is_key: bool
    is_geometry: bool


class SqlTable(Data):
    name: str
    key_column: Optional[SqlColumn]
    search_column: Optional[SqlColumn]
    geometry_column: Optional[SqlColumn]


class SqlSelectArgs(Data):
    columns: Optional[List[str]]
    extra_where: Optional[Sql]
    geometry_tolerance: Optional[float]
    keyword: Optional[str]
    limit: Optional[int]
    shape: Optional['IShape']
    sort: Optional[str]
    table: SqlTable
    uids: Optional[List[str]]


class IDbProvider(INode, Protocol):
    pass


class IDatabase(IDbProvider, Protocol):
    def select_features(self, args: 'SqlSelectArgs') -> List['IFeature']: ...

    def insert(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def update(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def delete(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def describe(self, table: 'SqlTable') -> Dict[str, 'SqlColumn']: ...


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
    planes: List['MapRenderInputPlane']
    rotation: int
    scale: int


class MapRenderOutputPlane(Data):
    type: Literal['image', 'path', 'svg']
    path: str
    elements: List[XmlElement]
    image: 'IImage'


class MapRenderOutput(Data):
    planes: List['MapRenderOutputPlane']
    view: MapView


class LayerRenderInput(Data):
    type: Literal['box', 'xyz', 'svg']
    view: MapView
    extraParams: dict
    x: int
    y: int
    z: int


class LayerRenderOutput(Data):
    content: bytes
    tags: List[XmlElement]


class TemplateRenderInputMap(Data):
    background_color: int
    bbox: Extent
    center: Point
    planes: List['MapRenderInputPlane']
    rotation: int
    scale: int
    visible_layers: Optional[List['ILayer']]


class TemplateRenderInput(Data):
    args: Optional[dict]
    crs: 'ICrs'
    dpi: Optional[int]
    locale_uid: Optional[str]
    maps: Optional[List[TemplateRenderInputMap]]
    out_mime: Optional[str]
    user: Optional['IUser']


class TemplateQualityLevel(Data):
    name: str
    dpi: int


class ITemplate(INode, Protocol):
    category: str
    data_model: Optional['IModel']
    mimes: List[str]
    name: str
    path: str
    subject: str
    text: str
    quality_levels: List[TemplateQualityLevel]

    map_size: Optional[MSize]
    page_size: Optional[MSize]

    def render(self, tri: TemplateRenderInput, notify: Callable = None) -> ContentResponse: ...


class ITemplateCollection(INode, Protocol):
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


class IFinder(INode, Protocol):
    data_model: Optional['IModel']

    supportsFilter: bool = False
    supportsGeometry: bool = False
    supportsKeyword: bool = False

    with_filter: bool
    with_geometry: bool
    with_keyword: bool

    cTemplates: Optional['ITemplateCollection']
    tolerance: 'Measurement'

    def run(self, args: SearchArgs, layer: 'ILayer' = None) -> List['IFeature']: ...

    def can_run(self, args: SearchArgs) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers


# noinspection PyPropertyDefinition
class IMap(INode, Protocol):
    rootLayer: 'ILayer'

    center: Point
    coordinatePrecision: int
    crs: 'ICrs'
    extent: Extent
    initResolution: float
    resolutions: List[float]


class LegendRenderOutput(Data):
    html: str
    image: 'IImage'
    image_path: str
    size: Size
    mime: str


class ILegend(INode, Protocol):
    def render(self, args: dict = None) -> Optional[LegendRenderOutput]: ...


class LayerDisplayMode(Enum):
    """Layer display mode"""

    box = 'box'  #: display a layer as one big image (WMS-alike)
    tile = 'tile'  #: display a layer in a tile grid
    client = 'client'  #: draw a layer in the client


# noinspection PyPropertyDefinition
class ILayer(INode, Protocol):
    canRenderBox: bool
    canRenderXyz: bool
    canRenderSvg: bool

    metadata: 'IMetadata'

    crs: 'ICrs'
    extent: Extent
    imageFormat: str
    opacity: float
    resolutions: List[float]

    hasCache: bool
    hasSearch: bool
    hasLegend: bool

    displayMode: LayerDisplayMode

    layers: List['ILayer']

    def own_bounds(self) -> Optional['Bounds']: ...

    def render(self, lri: LayerRenderInput) -> 'LayerRenderOutput': ...

    def render_legend(self, args: dict = None) -> Optional['LegendRenderOutput']: ...

    def render_description(self, args: dict = None) -> Optional[ContentResponse]: ...


#
# def render_xyz(self, x: int, y: int, z: int) -> bytes: ...
#
# def render_svg_element(self, view: 'MapView', style: Optional['IStyle']) -> Optional[XmlElement]: ...
#
# def render_svg_fragment(self, view: 'MapView', style: Optional['IStyle']) -> List[XmlElement]: ...

#
#
#
# is_group: bool
# is_editable: bool
#
# supports_raster_ows: bool
# supports_vector_ows: bool
#
# legend: 'Legend'
#
# display: str
#
# layers: List['ILayer'] = []
#
# templates: List['ITemplate']
# models: List['IModel']
# finders: List['IFinder']
#
#
# client_options: Data
#
# geometry_type: Optional[GeometryType]
# ows_enabled: bool
#
# def description(self) -> str: ...
#
# def has_search(self) -> bool: ...
#
# def has_legend(self) -> bool: ...
#
# def has_cache(self) -> bool: ...
#
# def own_bounds(self) -> Optional[Bounds]: ...
#
# def legend_url(self) -> Url: ...
#
# def ancestors(self) -> List['ILayer']: ...
#
#
# def render_legend_with_cache(self, context: dict = None) -> Optional[LegendRenderOutput]: ...
#
# def render_legend(self, context: dict = None) -> Optional[LegendRenderOutput]: ...
#
# def get_features(self, bounds: Bounds, limit: int = 0) -> List['IFeature']: ...


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
    cTemplates: 'ITemplateCollection'
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    def handle_request(self, req: 'IWebRequester') -> ContentResponse: ...


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
    sourceLayers: List['SourceLayer']


# ----------------------------------------------------------------------------------------------------------------------


# CLI

class CliParams(Data):
    """CLI params"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# actions and apis


class IAction(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# projects


class IClient(INode, Protocol):
    options: Dict
    elements: List


class IProject(INode, Protocol):
    assetsRoot: Optional['DocumentRoot']
    client: 'IClient'
    localeUids: List[str]
    map: 'IMap'
    metadata: 'IMetadata'

    def render_description(self, args: dict = None) -> Optional[ContentResponse]: ...


# ----------------------------------------------------------------------------------------------------------------------
# application

class IMonitor(INode, Protocol):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_path(self, path: str): ...

    def start(self): ...


class IApplication(INode, Protocol):
    auth: 'IAuthManager'
    client: 'IClient'
    localeUids: List[str]
    metadata: 'IMetadata'
    monitor: 'IMonitor'
    mpx_url: str
    webSites: List['IWebSite']
    qgisVersion: str
    version: str

    def developer_option(self, name: str): ...

    def find_project(self, uid: str) -> Optional['IProject']: ...

    def command_descriptor(
            self,
            command_category: str,
            command_name: str,
            params: dict,
            user: 'IUser',
            strict_mode: bool
    ) -> ExtCommandDescriptor: ...

    def actions_for(self, user: IGrantee, project: IProject = None) -> List['IAction']: ...

    def require_helper(self, ext_type: str): ...

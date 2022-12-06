from .data import Data

from gws.types import (
    cast,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

from gws.types import Enum

# ----------------------------------------------------------------------------------------------------------------------
# custom types, used everywhere


Extent = Tuple[float, float, float, float]
"""type: An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]."""

Point = Tuple[float, float]
"""type: Point coordinates [x, y]."""

Size = Tuple[float, float]
"""type: Size [width, height]."""

Corner = Literal['nw', 'sw', 'ne', 'se', 'lt', 'lb', 'rt', 'rb']
"""type: Corner."""


class Uom(Enum):
    """Unit of measure"""

    MI = 'mi'  # Statute mile EPSG:9093
    US_CH = 'us-ch'  # US survey chain EPSG:9033
    US_FT = 'us-ft'  # US survey foot EPSG:9003
    US_IN = 'us-in'  # US survey inch US_IN
    US_MI = 'us-mi'  # US survey mile EPSG:9035
    US_YD = 'us-yd'  # US survey yard US_YD
    CM = 'cm'  # centimetre EPSG:1033
    CH = 'ch'  # chain EPSG:9097
    DM = 'dm'  # decimeter DM
    DEG = 'deg'  # degree EPSG:9102
    FATH = 'fath'  # fathom EPSG:9014
    FT = 'ft'  # foot EPSG:9002
    GRAD = 'grad'  # grad EPSG:9105
    IN = 'in'  # inch IN
    KM = 'km'  # kilometre EPSG:9036
    LINK = 'link'  # link EPSG:9098
    M = 'm'  # metre EPSG:9001
    MM = 'mm'  # millimetre EPSG:1025
    KMI = 'kmi'  # nautical mile EPSG:9030
    RAD = 'rad'  # radian EPSG:9101
    YD = 'yd'  # yard EPSG:9096

    PX = 'px'  # pixel
    PT = 'pt'  # point


Measurement = Tuple[float, Uom]
"""type: A value with a unit."""

MPoint = Tuple[float, float, Uom]
"""type: A Point with a unit."""

MSize = Tuple[float, float, Uom]
"""type: A Size with a unit."""

Tag = tuple
"""type: An XML generator tag."""

FilePath = str
"""type: Valid readable file path on the server."""

DirPath = str
"""type: Valid readable directory path on the server."""

Duration = str
"""type: String like "1w 2d 3h 4m 5s" or a number of seconds."""

Color = str
"""type: CSS color name."""

Regex = str
"""type: Regular expression, as used in Python."""

FormatStr = str
"""type: String with {attribute} placeholders."""

Date = str
"""type: ISO date like "2019-01-30"."""

DateTime = str
"""type: ISO date/time like "2019-01-30 01:02:03"."""

Url = str
"""type: Http or https URL."""


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

    uid: str = ''
    """unique ID"""


Access = List[Tuple[int, str]]
"""type: Access Control List."""

ACL = str
"""type: A string of comma-separated pairs 'allow <role>' or 'deny <role>'."""


class ConfigWithAccess(Config):
    access: Optional[ACL]
    """access rights"""


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

    def can_use(self, obj: Any, *context) -> bool: ...

    def access_to(self, obj: Any) -> Optional[int]: ...


class INode(IObject, Protocol):
    config: Config
    root: 'IRoot'
    parent: 'INode'
    children: List['INode']

    def activate(self): ...

    def configure(self): ...

    def create_child(self, classref: ClassRef, config=None): ...

    def create_child_if_configured(self, classref: ClassRef, config=None): ...

    def create_children(self, classref: ClassRef, configs: Any): ...

    def post_configure(self): ...

    def pre_configure(self): ...

    def var(self, key: str, default=None): ...


class IRoot(Protocol):
    app: 'IApplication'
    specs: 'ISpecRuntime'
    configErrors: List[Any]

    def post_initialize(self): ...

    def activate(self): ...

    def find_all(self, classref: ClassRef) -> List: ...

    def find(self, classref: ClassRef, uid: str): ...

    def get(self, uid: str): ...

    def create(self, classref: ClassRef, parent: 'INode' = None, config=None): ...

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

    def read(self, value: Any, type_name: str, path: str = '', options=None) -> Any: ...

    def object_descriptor(self, type_name: str) -> Optional[ExtObjectDescriptor]: ...

    def command_descriptor(self, command_category: str, command_name: str) -> Optional[ExtCommandDescriptor]: ...

    def get_class(self, classref: ClassRef, ext_type: str = None) -> Optional[type]: ...

    def cli_docs(self, lang: str = 'en') -> dict: ...

    def bundle_paths(self, category: str) -> List[str]: ...

    def parse_classref(self, classref: ClassRef) -> Tuple[Optional[type], str, str]: ...


# ----------------------------------------------------------------------------------------------------------------------
# requests and responses

class Request(Data):
    """Web request"""

    projectUid: Optional[str]
    """project uid"""
    localeUid: Optional[str]
    """locale for this request"""


class EmptyRequest(Data):
    """Empty web request"""
    pass


class ResponseError(Data):
    """Web response error"""
    code: Optional[int]
    info: Optional[str]


class Response(Data):
    """Web response"""

    error: Optional[ResponseError]
    status: int


class ContentResponse(Response):
    """Web response with literal content"""

    attachment: Union[bool, str]
    content: Union[bytes, str]
    location: str
    mime: str
    path: str


class RequestMethod(Enum):
    GET = 'GET'
    HEAD = 'HEAD'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'
    CONNECT = 'CONNECT'
    OPTIONS = 'OPTIONS'
    TRACE = 'TRACE'
    PATCH = 'PATCH'


class IWebRequester(Protocol):
    environ: dict
    method: RequestMethod
    root: 'IRoot'
    site: 'IWebSite'
    session: 'IAuthSession'
    user: 'IUser'
    params: dict

    isApi: bool
    isGet: bool
    isPost: bool
    isSecure: bool

    def apply_middleware(self) -> 'IWebResponder': ...

    def cookie(self, key: str, default: str = '') -> str: ...

    def data(self) -> Optional[bytes]: ...

    def env(self, key: str, default: str = '') -> str: ...

    def has_param(self, key: str) -> bool: ...

    def header(self, key: str, default: str = '') -> str: ...

    def param(self, key: str, default: str = '') -> str: ...

    def text(self) -> Optional[str]: ...

    def content_responder(self, response: ContentResponse) -> 'IWebResponder': ...

    def struct_responder(self, response: Response) -> 'IWebResponder': ...

    def error_responder(self, exc: Exception) -> 'IWebResponder': ...

    def url_for(self, path: str, **params) -> Url: ...

    def find(self, klass, uid: str): ...

    def require(self, classref: ClassRef, uid: str): ...

    def require_project(self, uid: str) -> 'IProject': ...

    def require_layer(self, uid: str) -> 'ILayer': ...

    def acquire(self, classref: ClassRef, uid: str): ...


class IWebResponder(Protocol):
    status: int

    def send_response(self, environ, start_response): ...

    def set_cookie(self, key: str, **kwargs): ...

    def delete_cookie(self, key: str, **kwargs): ...

    def add_header(self, key: str, value): ...


# ----------------------------------------------------------------------------------------------------------------------
# web sites


class WebDocumentRoot(Data):
    dir: DirPath
    allowMime: Optional[List[str]]
    denyMime: Optional[List[str]]


class WebRewriteRule(Data):
    pattern: Regex
    target: str
    options: dict
    reversed: bool


class IWebManager(INode, Protocol):
    sites: List['IWebSite']

    def site_from_environ(self, environ: dict) -> 'IWebSite': ...


class IWebSite(INode, Protocol):
    assetsRoot: Optional[WebDocumentRoot]
    corsOptions: Data
    errorPage: Optional['ITemplate']
    host: str
    rewriteRules: List[WebRewriteRule]
    staticRoot: WebDocumentRoot

    def url_for(self, req: 'IWebRequester', path: str, **params) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization


class AuthPendingMfa(Data):
    status: str
    attemptCount: int
    restartCount: int
    timeStarted: int
    methodUid: str
    roles: Set[str]
    form: dict
    secret: Optional[str]


class IAuthManager(INode, Protocol):
    guestUser: 'IUser'
    systemUser: 'IUser'

    providers: List['IAuthProvider']
    methods: List['IAuthMethod']
    mfa: List['IAuthMfa']

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def get_user(self, user_uid: str) -> Optional['IUser']: ...

    def get_provider(self, uid: str) -> Optional['IAuthProvider']: ...

    def get_method(self, uid: str) -> Optional['IAuthMethod']: ...

    def get_mfa(self, uid: str) -> Optional['IAuthMfa']: ...

    def web_login(self, req: IWebRequester, credentials: Data): ...

    def web_logout(self, ): ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...

    def session_activate(self, req: IWebRequester, sess: Optional['IAuthSession']): ...

    def session_find(self, uid: str) -> Optional['IAuthSession']: ...

    def session_create(self, typ: str, method: 'IAuthMethod', user: 'IUser') -> 'IAuthSession': ...

    def session_save(self, sess: 'IAuthSession'): ...

    def session_delete(self, sess: 'IAuthSession'): ...

    def session_delete_all(self): ...


class IAuthMethod(INode, Protocol):
    secure: bool
    authMgr: 'IAuthManager'

    def open_session(self, req: 'IWebRequester') -> bool: ...

    def close_session(self, req: IWebRequester, res: IWebResponder) -> bool: ...


class IAuthProvider(INode, Protocol):
    allowedMethods: List[str]
    authMgr: 'IAuthManager'

    def get_user(self, local_uid: str) -> Optional['IUser']: ...

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, data: str) -> Optional['IUser']: ...


class IAuthMfa(INode, Protocol):
    autoStart: bool
    lifeTime: int
    maxAttempts: int
    maxRestarts: int

    def start(self, user: 'IUser'): ...

    def is_valid(self, user: 'IUser') -> bool: ...

    def cancel(self, user: 'IUser'): ...

    def verify(self, user: 'IUser', request: Data) -> bool: ...

    def restart(self, user: 'IUser') -> bool: ...


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
    localUid: str
    loginName: str
    pendingMfa: Optional[AuthPendingMfa]
    provider: 'IAuthProvider'
    uid: str


# ----------------------------------------------------------------------------------------------------------------------
# attributes and models

class AttributeType(Enum):
    BOOL = 'BOOL'
    BYTES = 'BYTES'
    DATE = 'DATE'
    DATETIME = 'DATETIME'
    FLOAT = 'FLOAT'
    FLOATLIST = 'FLOATLIST'
    GEOMETRY = 'GEOMETRY'
    INT = 'INT'
    INTLIST = 'INTLIST'
    STR = 'STR'
    STRLIST = 'STRLIST'
    TEXT = 'TEXT'
    TIME = 'TIME'


class GeometryType(Enum):
    """Geometry types: OGC and SQL/MM."""

    # sources:
    # OGC 06-103r4 (https://www.ogc.org/standards/sfa)
    # https://postgis.net/docs/manual-3.3/using_postgis_dbmanagement.htm

    GEOMETRY = 'GEOMETRY'

    POINT = 'POINT'
    CURVE = 'CURVE'
    SURFACE = 'SURFACE'

    GEOMETRYCOLLECTION = 'GEOMETRYCOLLECTION'

    LINESTRING = 'LINESTRING'
    LINE = 'LINE'
    LINEARRING = 'LINEARRING'

    POLYGON = 'POLYGON'
    TRIANGLE = 'TRIANGLE'

    POLYHEDRALSURFACE = 'POLYHEDRALSURFACE'
    TIN = 'TIN'

    MULTIPOINT = 'MULTIPOINT'
    MULTICURVE = 'MULTICURVE'
    MULTILINESTRING = 'MULTILINESTRING'
    MULTIPOLYGON = 'MULTIPOLYGON'
    MULTISURFACE = 'MULTISURFACE'

    CIRCULARSTRING = 'CIRCULARSTRING'
    COMPOUNDCURVE = 'COMPOUNDCURVE'
    CURVEPOLYGON = 'CURVEPOLYGON'


class IModel(INode, Protocol):
    def feature_from_source(self, sf: 'SourceFeature') -> 'IFeature': ...


class IModelManager(INode, Protocol):
    models: List['IModel']

    def add_model(self, m: IModel): ...

    def get_model_for(self, user: IGrantee = None, **kwargs) -> Optional[IModel]: ...


# ----------------------------------------------------------------------------------------------------------------------
# CRS

CrsName = Union[str, int]
"""type: CRS code like "EPSG:3857" or a srid like 3857."""


class CrsFormat(Enum):
    NONE = ''
    CRS = 'CRS'
    SRID = 'SRID'
    EPSG = 'EPSG'
    URL = 'URL'
    URI = 'URI'
    URNX = 'URNX'
    URN = 'URN'


class Axis(Enum):
    XY = 'XY'
    YX = 'YX'


class Bounds(Data):
    crs: 'ICrs'
    extent: Extent


class ICrs(Protocol):
    srid: str
    axis: Axis
    uom: Uom
    isGeographic: bool
    isProjected: bool
    isYX: bool
    proj4text: str
    wkt: str

    epsg: str
    urn: str
    urnx: str
    url: str
    uri: str

    name: str
    base: int
    datum: str

    wgsExtent: Extent
    extent: Extent

    def transform_extent(self, extent: Extent, crs_to: 'ICrs') -> Extent: ...

    def transformer(self, crs_to: 'ICrs') -> Callable: ...

    def to_string(self, fmt: 'CrsFormat' = None) -> str: ...

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
    tileWidth: float
    tileHeight: float
    extent: Extent


class TileMatrixSet(Data):
    uid: str
    crs: 'ICrs'
    matrices: List[TileMatrix]


class SourceStyle(Data):
    isDefault: bool
    legendUrl: Url
    metadata: 'Metadata'
    name: str


class SourceLayer(Data):
    aLevel: int
    aPath: str
    aUid: str

    dataSource: dict
    metadata: 'Metadata'

    supportedCrs: List['ICrs']
    wgsBounds: Bounds

    isExpanded: bool
    isGroup: bool
    isImage: bool
    isQueryable: bool
    isVisible: bool

    layers: List['SourceLayer']

    name: str
    title: str

    legendUrl: Url
    opacity: int
    scaleRange: List[float]

    styles: List[SourceStyle]
    defaultStyle: Optional[SourceStyle]

    tileMatrixIds: List[str]
    tileMatrixSets: List[TileMatrixSet]
    imageFormat: str
    resourceUrls: dict


class SourceFeature(Data):
    attributes: dict
    shape: Optional['IShape']
    layerName: Optional[str]
    uid: str


# ----------------------------------------------------------------------------------------------------------------------
# XML

class IXmlElement(Iterable):
    # ElementTree API

    tag: str
    """Tag name, with an optional namespace in Clark notation."""

    text: Optional[str]
    """Text before first subelement."""

    tail: Optional[str]
    """Text after this element's end tag."""

    attrib: dict
    """Dictionary of element attributes."""

    name: str
    """Element name (tag without a namespace)."""

    lname: str
    """Element name (tag without a namespace) in lower case."""

    caseInsensitive: bool
    """Elemet is case-insensitive."""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator['IXmlElement']: ...

    def __getitem__(self, item: int) -> 'IXmlElement': ...

    def clear(self): ...

    def get(self, key: str, default=None) -> Any: ...

    def attr(self, key: str, default=None) -> Any: ...

    def items(self) -> Iterable[Any]: ...

    def keys(self) -> Iterable[str]: ...

    def set(self, key: str, value: Any): ...

    def append(self, subelement: 'IXmlElement'): ...

    def extend(self, subelements: Iterable['IXmlElement']): ...

    def insert(self, index: int, subelement: 'IXmlElement'): ...

    def find(self, path: str) -> Optional['IXmlElement']: ...

    def findall(self, path: str) -> List['IXmlElement']: ...

    def findtext(self, path: str, default: str = None) -> str: ...

    def iter(self, tag: str = None) -> Iterable['IXmlElement']: ...

    def iterfind(self, path: str = None) -> Iterable['IXmlElement']: ...

    def itertext(self) -> Iterable[str]: ...

    # extensions

    def children(self) -> List['IXmlElement']: ...

    def findfirst(self, *paths) -> Optional['IXmlElement']: ...

    def textof(self, *paths) -> str: ...

    def textlist(self, *paths, deep=False) -> List[str]: ...

    def textdict(self, *paths, deep=False) -> Dict[str, str]: ...

    #

    def to_string(
            self,
            compact_whitespace=False,
            remove_namespaces=False,
            with_namespace_declarations=False,
            with_schema_locations=False,
            with_xml_declaration=False,
    ) -> str: ...

    def to_dict(self) -> dict: ...


# ----------------------------------------------------------------------------------------------------------------------
# shapes

class IShape(IObject, Protocol):
    """Georeferenced geometry."""

    type: GeometryType
    """Geometry type."""

    crs: 'ICrs'
    """CRS of this shape."""

    x: Optional[float]
    """X-coordinate for Point geometries, None otherwise."""

    y: Optional[float]
    """Y-coordinate for Point geometries, None otherwise."""

    # common props

    def area(self) -> float:
        """Computes the area of the geometry."""

    def bounds(self) -> Bounds:
        """Retuns a Bounds object that bounds this shape."""

    def centroid(self) -> 'IShape':
        """Returns a centroid as a Point shape."""

    # formats

    def to_wkb(self) -> bytes:
        """Returns a WKB representation of this shape as a binary string."""

    def to_wkb_hex(self) -> str:
        """Returns a WKB representation of this shape as a hex string."""

    def to_ewkb(self) -> bytes:
        """Returns an EWKB representation of this shape as a binary string."""

    def to_ewkb_hex(self) -> str:
        """Returns an EWKB representation of this shape as a hex string."""

    def to_wkt(self) -> str:
        """Returns a WKT representation of this shape."""

    def to_ewkt(self) -> str:
        """Returns an EWKT representation of this shape."""

    def to_geojson(self, always_xy=False) -> dict:
        """Returns a GeoJSON representation of this shape."""

    # predicates (https://shapely.readthedocs.io/en/stable/manual.html#predicates-and-relationships)

    def is_empty(self) -> bool:
        """Returns True if this shape is empty."""

    def is_ring(self) -> bool:
        """Returns True if this shape is a ring."""

    def is_simple(self) -> bool:
        """Returns True if this shape is 'simple'."""

    def is_valid(self) -> bool:
        """Returns True if this shape is valid."""

    def equals(self, other: 'IShape') -> bool:
        """Returns True if this shape is equal to the other."""

    def contains(self, other: 'IShape') -> bool:
        """Returns True if this shape contains the other."""

    def covers(self, other: 'IShape') -> bool:
        """Returns True if this shape covers the other."""

    def covered_by(self, other: 'IShape') -> bool:
        """Returns True if this shape is covered by the other."""

    def crosses(self, other: 'IShape') -> bool:
        """Returns True if this shape crosses the other."""

    def disjoint(self, other: 'IShape') -> bool:
        """Returns True if this shape does not intersect with the other."""

    def intersects(self, other: 'IShape') -> bool:
        """Returns True if this shape intersects with the other."""

    def overlaps(self, other: 'IShape') -> bool:
        """Returns True if this shape overlaps the other."""

    def touches(self, other: 'IShape') -> bool:
        """Returns True if this shape touches the other."""

    def within(self, other: 'IShape') -> bool:
        """Returns True if this shape is within the other."""

    # set operations

    def union(self, *others: 'IShape') -> 'IShape':
        """Computes a union of this shape and other shapes."""

    def intersection(self, *others: 'IShape') -> 'IShape':
        """Computes an intersection of this shape and other shapes."""

    # convertors

    def to_multi(self) -> 'IShape':
        """Converts a singly-geometry shape to a multi-geometry one."""

    def to_type(self, new_type: 'GeometryType') -> 'IShape':
        """Converts a geometry to another type."""

    # misc

    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape':
        """Builds a buffer polygon around the shape."""

    def transformed_to(self, crs: 'ICrs') -> 'IShape':
        """Returns this shape transormed to another CRS."""


# ----------------------------------------------------------------------------------------------------------------------
# features

# noinspection PyPropertyDefinition
class IFeature(IObject, Protocol):
    attributes: dict
    elements: dict

    shape: Optional['IShape']
    layerName: Optional[str]
    uid: str

    model: 'IModel'
    isNew: bool

    def apply_template(self, template: 'ITemplate', extra_args: dict = None) -> 'IFeature': ...

    def attr(self, name: str, default=None) -> Any: ...

    def to_geojson(self) -> dict: ...

    def to_svg_fragment(self, view: 'MapView', style: 'IStyle' = None) -> List[IXmlElement]: ...

    def transform_to(self, crs: 'ICrs') -> 'IFeature': ...


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


class IDatabaseManager(INode, Protocol):
    databases: List['IDatabase']


class IDatabase(INode, Protocol):
    def select_features(self, args: 'SqlSelectArgs') -> List['IFeature']: ...

    def insert(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def update(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def delete(self, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: ...

    def describe(self, table: 'SqlTable') -> Dict[str, 'SqlColumn']: ...


# ----------------------------------------------------------------------------------------------------------------------
# templates and rendering

class IImage(IObject, Protocol):
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
    elements: List[IXmlElement]
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
    tags: List[IXmlElement]


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
    localeUid: Optional[str]
    maps: Optional[List[TemplateRenderInputMap]]
    mimeOut: Optional[str]
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


class ITemplateManager(INode, Protocol):
    templates: List['ITemplate']

    def get_template_for(self, user: IGrantee = None, **kwargs) -> Optional['ITemplate']: ...

    def render_template(self, tri: TemplateRenderInput, user: IGrantee = None, **kwargs) -> Optional[ContentResponse]: ...


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
    dateUnits: str
    """date unit names, e.g. 'YMD' for 'en', 'JMT' for 'de'"""
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

    description: str
    format: str
    formatVersion: str
    function: str
    mimeType: str
    scheme: str
    title: str
    type: str
    url: Url


class MetadataAccessConstraint(Data):
    title: str
    type: str


class MetadataLicense(Data):
    title: str
    url: Url


class MetadataAttribution(Data):
    title: str
    url: Url


class Metadata(Data):
    abstract: str
    accessConstraints: List[MetadataAccessConstraint]
    attribution: MetadataAttribution
    authorityIdentifier: str
    authorityName: str
    authorityUrl: str
    catalogCitationUid: str
    catalogUid: str
    fees: str
    image: str
    keywords: List[str]
    language3: str
    language: str
    languageName: str
    license: MetadataLicense
    name: str
    parentIdentifier: str
    title: str

    contactAddress: str
    contactAddressType: str
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
    isoTopicCategories: List[str]
    isoSpatialResolution: str

    metaLinks: List[MetadataLink]
    serviceMetaLink: MetadataLink
    extraLinks: List[MetadataLink]


# ----------------------------------------------------------------------------------------------------------------------
# search

class SearchFilter(Data):
    name: str
    operator: str
    shape: 'IShape'
    sub: List['SearchFilter']
    value: str


class SearchArgs(Data):
    bounds: Bounds
    keyword: Optional[str]
    filter: Optional['SearchFilter']
    layers: List['ILayer']
    limit: int
    params: dict
    project: 'IProject'
    resolution: float
    shapes: List['IShape']
    tolerance: 'Measurement'
    user: IGrantee
    featureElements: Optional[List[str]]


class IFinder(INode, Protocol):
    supportsFilter: bool
    supportsGeometry: bool
    supportsKeyword: bool

    withFilter: bool
    withGeometry: bool
    withKeyword: bool

    templateMgr: Optional['ITemplateManager']
    modelMgr: Optional['IModelManager']

    tolerance: 'Measurement'

    def run(self, args: SearchArgs, layer: 'ILayer') -> List['IFeature']: ...

    def can_run(self, args: SearchArgs) -> bool: ...


class ISearchManager(INode, Protocol):
    finders: List['IFinder']

    def add_finder(self, f: IFinder): ...

    def get_finder_for(self, user: IGrantee = None, **kwargs) -> Optional[IFinder]: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers


# noinspection PyPropertyDefinition
class IMap(INode, Protocol):
    rootLayer: 'ILayer'

    bounds: Bounds
    center: Point
    coordinatePrecision: int
    initResolution: float
    resolutions: List[float]
    title: str
    wgsExtent: Extent


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

    box = 'box'
    """display a layer as one big image (WMS-alike)"""
    tile = 'tile'
    """display a layer in a tile grid"""
    client = 'client'
    """draw a layer in the client"""


class TileGrid(Data):
    uid: str
    bounds: Bounds
    corner: Corner
    resolutions: List[float]
    tileSize: int


# noinspection PyPropertyDefinition
class ILayer(INode, Protocol):
    canRenderBox: bool
    canRenderXyz: bool
    canRenderSvg: bool

    supportsRasterServices: bool
    supportsVectorServices: bool

    bounds: Bounds
    displayMode: LayerDisplayMode
    imageFormat: str
    opacity: float
    resolutions: List[float]
    title: str

    sourceGrid: Optional[TileGrid]
    targetGrid: Optional[TileGrid]

    metadata: 'Metadata'
    legend: Optional['ILegend']
    searchMgr: 'ISearchManager'
    templateMgr: 'ITemplateManager'
    modelMgr: 'IModelManager'

    layers: List['ILayer']

    def ancestors(self) -> List['ILayer']: ...

    def render(self, lri: LayerRenderInput) -> 'LayerRenderOutput': ...

    def render_legend(self, args: dict = None) -> Optional['LegendRenderOutput']: ...

    def url_path(self, kind: Literal['box', 'tile', 'legend', 'features']) -> str: ...


#
# def render_xyz(self, x: int, y: int, z: int) -> bytes: ...
#
# def render_svg_element(self, view: 'MapView', style: Optional['IStyle']) -> Optional[IXmlElement]: ...
#
# def render_svg_fragment(self, view: 'MapView', style: Optional['IStyle']) -> List[IXmlElement]: ...

#
#
#
# isGroup: bool
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
# def legendUrl(self) -> Url: ...
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
    allowedParameters: Dict[str, List[str]]
    constraints: Dict[str, List[str]]
    formats: List[str]
    params: Dict[str, str]
    postUrl: Url
    preferredFormat: str
    url: Url
    verb: OwsVerb


class IOwsService(INode, Protocol):
    metadata: 'Metadata'
    name: str
    protocol: OwsProtocol
    supported_bounds: List[Bounds]
    supported_versions: List[str]
    templateMgr: 'ITemplateManager'
    version: str
    with_inspire_meta: bool
    with_strict_params: bool

    def handle_request(self, req: 'IWebRequester') -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    forceCrs: 'ICrs'
    alwaysXY: bool
    metadata: 'Metadata'
    operations: List[OwsOperation]
    protocol: OwsProtocol
    sourceLayers: List['SourceLayer']
    url: Url
    version: str

    def get_operation(self, verb: OwsVerb, method: RequestMethod = None) -> Optional[OwsOperation]: ...


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


class IActionManager(INode, Protocol):
    items: List['IAction']

    def find_action(self, class_name: str) -> Optional['IAction']: ...

    def actions_for(self, user: IGrantee, other: 'IActionManager' = None) -> List['IAction']: ...


class IAction(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# projects


class IClient(INode, Protocol):
    options: Dict
    elements: List


class IProject(INode, Protocol):
    actionMgr: 'IActionManager'
    assetsRoot: Optional['WebDocumentRoot']
    client: 'IClient'
    localeUids: List[str]
    map: 'IMap'
    metadata: 'Metadata'
    templateMgr: 'ITemplateManager'


# ----------------------------------------------------------------------------------------------------------------------
# application

class IMonitor(INode, Protocol):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_path(self, path: str): ...

    def start(self): ...


WebMiddlewareHandler = Callable[['IWebRequester', Callable], 'IWebResponder']


class IApplication(INode, Protocol):
    client: 'IClient'
    localeUids: List[str]
    metadata: 'Metadata'
    monitor: 'IMonitor'
    qgisVersion: str
    version: str
    versionString: str

    actionMgr: 'IActionManager'
    authMgr: 'IAuthManager'
    databaseMgr: 'IDatabaseManager'
    webMgr: 'IWebManager'

    def register_web_middleware(self, name: str, fn: WebMiddlewareHandler): ...

    def web_middleware_list(self) -> List[WebMiddlewareHandler]: ...

    def developer_option(self, name: str): ...

    def get_project(self, uid: str) -> Optional['IProject']: ...

    def require_helper(self, ext_type: str): ...

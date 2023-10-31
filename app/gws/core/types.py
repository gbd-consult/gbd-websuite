"""Basic GWS types."""

# mypy: disable-error-code="empty-body"


from .data import Data

from gws.types import (
    cast,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
)

from gws.types import Enum

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime
    import sqlalchemy
    import sqlalchemy.orm

# ----------------------------------------------------------------------------------------------------------------------
# custom types, used everywhere

Extent: TypeAlias = tuple[float, float, float, float]
"""An array of 4 elements representing extent coordinates ``[minx, miny, maxx, maxy]``."""

Point: TypeAlias = tuple[float, float]
"""Point coordinates ``[x, y]``."""

Size: TypeAlias = tuple[float, float]
"""Size ``[width, height]``."""


class Origin(Enum):
    """Grid origin."""
    nw = 'nw'
    """north-west"""
    sw = 'sw'
    """south-west"""
    ne = 'ne'
    """north-east"""
    se = 'se'
    """south-east"""
    lt = 'nw'
    """left top"""
    lb = 'sw'
    """left bottom"""
    rt = 'ne'
    """right top"""
    rb = 'se'
    """right bottom"""


class Uom(Enum):
    """Unit of measure"""

    mi = 'mi'
    """statute mile epsg:9093"""
    us_ch = 'us-ch'
    """us survey chain epsg:9033"""
    us_ft = 'us-ft'
    """us survey foot epsg:9003"""
    us_in = 'us-in'
    """us survey inch us_in"""
    us_mi = 'us-mi'
    """us survey mile epsg:9035"""
    us_yd = 'us-yd'
    """us survey yard us_yd"""
    cm = 'cm'
    """centimetre epsg:1033"""
    ch = 'ch'
    """chain epsg:9097"""
    dm = 'dm'
    """decimeter dm"""
    deg = 'deg'
    """degree epsg:9102"""
    fath = 'fath'
    """fathom epsg:9014"""
    ft = 'ft'
    """foot epsg:9002"""
    grad = 'grad'
    """grad epsg:9105"""
    inch = 'in'
    """inch in"""
    km = 'km'
    """kilometre epsg:9036"""
    link = 'link'
    """link epsg:9098"""
    m = 'm'
    """metre epsg:9001"""
    mm = 'mm'
    """millimetre epsg:1025"""
    kmi = 'kmi'
    """nautical mile epsg:9030"""
    rad = 'rad'
    """radian epsg:9101"""
    yd = 'yd'
    """yard epsg:9096"""
    px = 'px'
    """pixel"""
    pt = 'pt'
    """point"""


Measurement: TypeAlias = tuple[float, Uom]
"""A value with a unit, like ``"5mm"``."""

MPoint: TypeAlias = tuple[float, float, Uom]
"""A Point with a unit."""

MSize: TypeAlias = tuple[float, float, Uom]
"""A Size with a unit, like ``["1mm", "2mm"]``."""

MExtent: TypeAlias = tuple[float, float, float, float, Uom]
"""An Extent with a unit, like ``["1mm", "2mm", "3mm", "4mm"]``"""

Tag: TypeAlias = tuple
"""An XML generator tag."""

FilePath: TypeAlias = str
"""Valid readable file path on the server."""

DirPath: TypeAlias = str
"""Valid readable directory path on the server."""

Duration: TypeAlias = str
"""A string like ``1w 2d 3h 4m 5s`` or an integer number of seconds."""

Color: TypeAlias = str
"""A CSS color name."""

Regex: TypeAlias = str
"""A regular expression, as used in Python."""

FormatStr: TypeAlias = str
"""A string with ``{attribute}`` placeholders."""

Date: TypeAlias = str
"""An ISO date string like ``2019-01-30``."""

DateTime: TypeAlias = str
"""An ISO datetime string like ``2019-01-30 01:02:03``."""

Url: TypeAlias = str
"""An http or https URL."""


# ----------------------------------------------------------------------------------------------------------------------
# application manifest


class ApplicationManifestPlugin(Data):
    path: FilePath
    name: str = ''


class ApplicationManifest(Data):
    excludePlugins: Optional[list[str]]
    plugins: Optional[list[ApplicationManifestPlugin]]
    locales: list[str]

    withFallbackConfig: bool = False
    withStrictConfig: bool = False


# ----------------------------------------------------------------------------------------------------------------------
# basic objects

ClassRef: TypeAlias = type | str


class Config(Data):
    """Configuration base type"""

    uid: str = ''
    """unique ID"""


class Props(Data):
    """Properties base type"""
    uid: str = ''
    """unique ID"""


# ----------------------------------------------------------------------------------------------------------------------
# permissions


Acl: TypeAlias = list[tuple[int, str]]
"""Access Control list."""

AclSpec: TypeAlias = str
"""A string of comma-separated pairs ``allow <role>`` or ``deny <role>``."""


class Access(Enum):
    """Access mode."""
    read = 'read'
    write = 'write'
    create = 'create'
    delete = 'delete'


class AccessConfig:
    """Access permissions."""
    read: Optional[AclSpec]
    """permission to read the object"""
    write: Optional[AclSpec]
    """permission to change the object"""
    create: Optional[AclSpec]
    """permission to create new objects"""
    delete: Optional[AclSpec]
    """permission to delete objects"""
    edit: Optional[AclSpec]
    """read, write, create and delete"""


class ConfigWithAccess(Config):
    access: Optional[AclSpec]
    """permission to use the object"""
    permissions: Optional[AccessConfig]
    """additional permissions"""


# ----------------------------------------------------------------------------------------------------------------------
# foundation interfaces

class IObject(Protocol):
    permissions: dict[Access, Acl]

    def props(self, user: 'IUser') -> Props: ...


class INode(IObject, Protocol):
    extName: str
    extType: str

    config: Config
    root: 'IRoot'
    parent: 'INode'
    children: list['INode']
    uid: str

    def activate(self): ...

    def configure(self): ...

    def create_child(self, classref: ClassRef, config=None, **kwargs): ...

    def create_child_if_configured(self, classref: ClassRef, config=None, **kwargs): ...

    def create_children(self, classref: ClassRef, configs: list, **kwargs): ...

    def post_configure(self): ...

    def pre_configure(self): ...

    def cfg(self, key: str, default=None): ...

    def find_all(self, classref: ClassRef) -> list['INode']: ...

    def find_first(self, classref: ClassRef) -> Optional['INode']: ...

    def closest(self, classref: ClassRef) -> Optional['INode']: ...


class IRoot(Protocol):
    app: 'IApplication'
    specs: 'ISpecRuntime'
    configErrors: list[Any]

    def post_initialize(self): ...

    def activate(self): ...

    def find_all(self, classref: ClassRef) -> list[INode]: ...

    def find_first(self, classref: ClassRef) -> Optional[INode]: ...

    def get(self, uid: str, classref: Optional[ClassRef] = None) -> Optional[INode]: ...

    def object_count(self) -> int: ...

    def create(self, classref: ClassRef, parent: Optional['INode'] = None, config=None, **kwargs): ...

    def create_shared(self, classref: ClassRef, config=None, **kwargs): ...

    def create_temporary(self, classref: ClassRef, config=None, **kwargs): ...

    def create_application(self, config=None, **kwargs) -> 'IApplication': ...


class IProvider(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# spec runtime


class ExtObjectDescriptor(Data):
    extName: str
    extType: str
    classPtr: type
    ident: str
    modName: str
    modPath: str


class ExtCommandDescriptor(Data):
    extName: str
    extType: str
    methodName: str
    methodPtr: Callable
    request: 'Request'
    tArg: str
    tOwner: str
    owner: ExtObjectDescriptor


class ISpecRuntime(Protocol):
    version: str
    manifest: ApplicationManifest

    def read(self, value: Any, type_name: str, path: str = '', options=None) -> Any: ...

    def object_descriptor(self, type_name: str) -> Optional[ExtObjectDescriptor]: ...

    def command_descriptor(self, command_category: str, command_name: str) -> Optional[ExtCommandDescriptor]: ...

    def get_class(self, classref: ClassRef, ext_type: Optional[str] = None) -> Optional[type]: ...

    def cli_docs(self, lang: str = 'en') -> dict: ...

    def bundle_paths(self, category: str) -> list[str]: ...

    def parse_classref(self, classref: ClassRef) -> tuple[Optional[type], str, str]: ...


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

    asAttachment: bool
    attachmentName: str
    content: bytes | str
    location: str
    mime: str
    path: str
    headers: dict


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
    params: dict
    command: str

    session: 'IAuthSession'
    user: 'IUser'

    isApi: bool
    isGet: bool
    isPost: bool
    isSecure: bool

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

    def require(self, uid: str, classref: ClassRef): ...

    def require_project(self, uid: str) -> 'IProject': ...

    def require_layer(self, uid: str) -> 'ILayer': ...

    def acquire(self, uid: str, classref: ClassRef): ...

    def set_session(self, sess: 'IAuthSession'): ...


class IWebResponder(Protocol):
    status: int

    def send_response(self, environ, start_response): ...

    def set_cookie(self, key: str, **kwargs): ...

    def delete_cookie(self, key: str, **kwargs): ...

    def set_status(self, value): ...

    def add_header(self, key: str, value): ...


# ----------------------------------------------------------------------------------------------------------------------
# web sites


class WebDocumentRoot(Data):
    dir: DirPath
    allowMime: Optional[list[str]]
    denyMime: Optional[list[str]]


class WebRewriteRule(Data):
    pattern: Regex
    target: str
    options: dict
    reversed: bool


class IWebManager(INode, Protocol):
    sites: list['IWebSite']

    def site_from_environ(self, environ: dict) -> 'IWebSite': ...


class IWebSite(INode, Protocol):
    assetsRoot: Optional[WebDocumentRoot]
    corsOptions: Data
    errorPage: Optional['ITemplate']
    host: str
    rewriteRules: list[WebRewriteRule]
    staticRoot: WebDocumentRoot

    def url_for(self, req: 'IWebRequester', path: str, **params) -> Url: ...


# ----------------------------------------------------------------------------------------------------------------------
# authorization


class IUser(IObject, Protocol):
    attributes: dict[str, Any]
    displayName: str
    isGuest: bool
    localUid: str
    loginName: str
    provider: 'IAuthProvider'
    roles: set[str]
    uid: str

    def acl_bit(self, access: Access, obj: IObject) -> Optional[int]: ...

    def can(self, access: Access, obj: IObject, *context) -> bool: ...

    def can_create(self, obj: IObject, *context) -> bool: ...

    def can_delete(self, obj: IObject, *context) -> bool: ...

    def can_read(self, obj: IObject, *context) -> bool: ...

    def can_use(self, obj: IObject, *context) -> bool: ...

    def can_write(self, obj: IObject, *context) -> bool: ...

    def can_edit(self, obj: IObject, *context) -> bool: ...

    def acquire(self, uid: str, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> Optional[IObject]: ...

    def require(self, uid: str, classref: Optional[ClassRef] = None, access: Optional[Access] = None) -> IObject: ...


class IAuthSession(IObject, Protocol):
    uid: str
    method: Optional['IAuthMethod']
    user: 'IUser'
    data: dict
    created: 'datetime.datetime'
    updated: 'datetime.datetime'
    isChanged: bool

    def get(self, key: str, default=None): ...

    def set(self, key: str, val: Any): ...


class IAuthManager(INode, Protocol):
    guestSession: 'IAuthSession'

    guestUser: 'IUser'
    systemUser: 'IUser'

    providers: list['IAuthProvider']
    methods: list['IAuthMethod']
    mfa: list['IAuthMfa']

    sessionMgr: 'IAuthSessionManager'

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def get_user(self, user_uid: str) -> Optional['IUser']: ...

    def get_provider(self, uid: str) -> Optional['IAuthProvider']: ...

    def get_method(self, uid: str) -> Optional['IAuthMethod']: ...

    def get_mfa(self, uid: str) -> Optional['IAuthMfa']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, ser: str) -> Optional['IUser']: ...


class IAuthMethod(INode, Protocol):
    authMgr: 'IAuthManager'
    secure: bool

    def open_session(self, req: IWebRequester) -> Optional['IAuthSession']: ...

    def close_session(self, req: IWebRequester, res: IWebResponder) -> bool: ...


class IAuthMfa(INode, Protocol):
    authMgr: 'IAuthManager'
    autoStart: bool
    lifeTime: int
    maxAttempts: int
    maxRestarts: int

    def start(self, user: 'IUser'): ...

    def is_valid(self, user: 'IUser') -> bool: ...

    def cancel(self, user: 'IUser'): ...

    def verify(self, user: 'IUser', request: Data) -> bool: ...

    def restart(self, user: 'IUser') -> bool: ...


class IAuthProvider(INode, Protocol):
    authMgr: 'IAuthManager'
    allowedMethods: list[str]

    def get_user(self, local_uid: str) -> Optional['IUser']: ...

    def authenticate(self, method: 'IAuthMethod', credentials: Data) -> Optional['IUser']: ...

    def serialize_user(self, user: 'IUser') -> str: ...

    def unserialize_user(self, data: str) -> Optional['IUser']: ...


class IAuthSessionManager(INode, Protocol):
    authMgr: 'IAuthManager'
    lifeTime: int

    def create(self, method: 'IAuthMethod', user: 'IUser', data: Optional[dict] = None) -> 'IAuthSession': ...

    def delete(self, sess: 'IAuthSession'): ...

    def delete_all(self): ...

    def get(self, uid: str) -> Optional['IAuthSession']: ...

    def get_valid(self, uid: str) -> Optional['IAuthSession']: ...

    def get_all(self) -> list['IAuthSession']: ...

    def save(self, sess: 'IAuthSession'): ...

    def touch(self, sess: 'IAuthSession'): ...

    def cleanup(self): ...


# ----------------------------------------------------------------------------------------------------------------------
# attributes

class AttributeType(Enum):
    """Feature attribute type."""

    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    feature = 'feature'
    featurelist = 'featurelist'
    file = 'file'
    float = 'float'
    floatlist = 'floatlist'
    geometry = 'geometry'
    int = 'int'
    intlist = 'intlist'
    str = 'str'
    strlist = 'strlist'
    time = 'time'


class GeometryType(Enum):
    """Feature geometry type.

    OGC and SQL/MM types.

    Sources:
    - OGC 06-103r4 (https://www.ogc.org/standards/sfa)
    - https://postgis.net/docs/manual-3.3/using_postgis_dbmanagement.htm
    """

    geometry = 'geometry'

    point = 'point'
    curve = 'curve'
    surface = 'surface'

    geometrycollection = 'geometrycollection'

    linestring = 'linestring'
    line = 'line'
    linearring = 'linearring'

    polygon = 'polygon'
    triangle = 'triangle'

    polyhedralsurface = 'polyhedralsurface'
    tin = 'tin'

    multipoint = 'multipoint'
    multicurve = 'multicurve'
    multilinestring = 'multilinestring'
    multipolygon = 'multipolygon'
    multisurface = 'multisurface'

    circularstring = 'circularstring'
    compoundcurve = 'compoundcurve'
    curvepolygon = 'curvepolygon'


# ----------------------------------------------------------------------------------------------------------------------
# CRS

CrsName: TypeAlias = int | str
"""A CRS code like `EPSG:3857` or a srid like `3857`."""


class CrsFormat(Enum):
    none = ''
    crs = 'crs'
    srid = 'srid'
    epsg = 'epsg'
    url = 'url'
    uri = 'uri'
    urnx = 'urnx'
    urn = 'urn'


class Axis(Enum):
    xy = 'xy'
    yx = 'yx'


class Bounds(Data):
    crs: 'ICrs'
    extent: Extent


class ICrs(Protocol):
    srid: int
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

    def to_string(self, fmt: Optional['CrsFormat'] = None) -> str: ...

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
    matrices: list[TileMatrix]


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

    supportedCrs: list['ICrs']
    wgsBounds: Bounds

    isExpanded: bool
    isGroup: bool
    isImage: bool
    isQueryable: bool
    isVisible: bool

    layers: list['SourceLayer']

    name: str
    title: str

    legendUrl: Url
    opacity: int
    scaleRange: list[float]

    styles: list[SourceStyle]
    defaultStyle: Optional[SourceStyle]

    tileMatrixIds: list[str]
    tileMatrixSets: list[TileMatrixSet]
    imageFormat: str
    resourceUrls: dict


# ----------------------------------------------------------------------------------------------------------------------
# XML


class XmlNamespace(Data):
    """XML namespace"""

    uid: str
    """unique ID"""
    xmlns: str
    """default namespace prefix"""
    uri: Url
    """namespace uri"""
    schemaLocation: Url
    """namespace schema location"""
    version: str
    """namespace version"""


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

    def findall(self, path: str) -> list['IXmlElement']: ...

    def findtext(self, path: str, default: Optional[str] = None) -> str: ...

    def iter(self, tag: Optional[str] = None) -> Iterable['IXmlElement']: ...

    def iterfind(self, path: Optional[str] = None) -> Iterable['IXmlElement']: ...

    def itertext(self) -> Iterable[str]: ...

    def remove(self, other: 'IXmlElement'): ...

    # extensions

    def add(self, tag: str, attrib: Optional[dict] = None, **extra) -> 'IXmlElement': ...

    def children(self) -> list['IXmlElement']: ...

    def findfirst(self, *paths) -> Optional['IXmlElement']: ...

    def textof(self, *paths) -> str: ...

    def textlist(self, *paths, deep=False) -> list[str]: ...

    def textdict(self, *paths, deep=False) -> dict[str, str]: ...

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

class ShapeProps(Props):
    """Shape properties object."""
    crs: str
    geometry: dict


class IShape(Protocol):
    """Geo-referenced geometry."""

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
        """Returns a Bounds object that bounds this shape."""

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

    def to_props(self) -> ShapeProps:
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

    def union(self, others: list['IShape']) -> 'IShape':
        """Computes a union of this shape and other shapes."""

    def intersection(self, *others: 'IShape') -> 'IShape':
        """Computes an intersection of this shape and other shapes."""

    # convertors

    def to_multi(self) -> 'IShape':
        """Converts a singly-geometry shape to a multi-geometry one."""

    def to_type(self, new_type: 'GeometryType') -> 'IShape':
        """Converts a geometry to another type."""

    # misc

    def tolerance_polygon(self, tolerance, quad_segs=None) -> 'IShape':
        """Builds a buffer polygon around the shape."""

    def transformed_to(self, crs: 'ICrs') -> 'IShape':
        """Returns this shape transformed to another CRS."""


# ----------------------------------------------------------------------------------------------------------------------
# database

class ColumnDescription(Data):
    columnIndex: int
    comment: str
    default: str
    geometrySrid: int
    geometryType: GeometryType
    isAutoincrement: bool
    isNullable: bool
    isPrimaryKey: bool
    isUnique: bool
    hasDefault: bool
    name: str
    nativeType: str
    options: dict
    type: AttributeType


class RelationshipDescription(Data):
    name: str
    schema: str
    fullName: str
    foreignKeys: str
    referredKeys: str


class DataSetDescription(Data):
    columns: list[ColumnDescription]
    columnMap: dict[str, ColumnDescription]
    fullName: str
    geometryName: str
    geometrySrid: int
    geometryType: GeometryType
    name: str
    schema: str


class IDatabaseManager(INode, Protocol):
    def create_provider(self, cfg: Config, **kwargs) -> 'IDatabaseProvider': ...

    def providers(self) -> list['IDatabaseProvider']: ...

    def provider(self, uid: str) -> Optional['IDatabaseProvider']: ...

    def first_provider(self, ext_type: str) -> Optional['IDatabaseProvider']: ...


class IDatabaseProvider(IProvider, Protocol):
    mgr: 'IDatabaseManager'
    url: str
    models: list['IDatabaseModel']

    def connection(self) -> 'sqlalchemy.Connection': ...

    def engine(self, **kwargs) -> 'sqlalchemy.Engine': ...

    def split_table_name(self, table_name: str) -> tuple[str, str]: ...

    def join_table_name(self, schema: str, name: str) -> str: ...

    def table(self, table_name: str, **kwargs) -> 'sqlalchemy.Table': ...

    def has_table(self, table_name: str) -> bool: ...

    def column(self, table: 'sqlalchemy.Table', column_name: str) -> 'sqlalchemy.Column': ...

    def has_column(self, table: 'sqlalchemy.Table', column_name: str) -> bool: ...

    def describe(self, table_name: str) -> DataSetDescription: ...

    def table_bounds(self, table_name) -> Optional[Bounds]: ...


# ----------------------------------------------------------------------------------------------------------------------
# storage

class IStorageManager(INode, Protocol):
    def provider(self, uid: str) -> Optional['IStorageProvider']: ...

    def first_provider(self) -> Optional['IStorageProvider']: ...


class StorageRecord(Data):
    name: str
    userUid: str
    data: str
    created: int
    updated: int


class IStorageProvider(INode, Protocol):
    def list_names(self, category: str) -> list[str]: ...

    def read(self, category: str, name: str) -> Optional['StorageRecord']: ...

    def write(self, category: str, name: str, data: str, user_uid: str): ...

    def delete(self, category: str, name: str): ...


# ----------------------------------------------------------------------------------------------------------------------
# features


FeatureUid: TypeAlias = str


class FeatureRecord(Data):
    """Raw data from a feature source."""
    attributes: dict
    meta: dict
    uid: Optional[str]
    shape: Optional['IShape']


class FeatureProps(Props):
    attributes: dict
    cssSelector: str
    errors: Optional[list['ModelValidationError']]
    createWithFeatures: Optional[list['FeatureProps']]
    isNew: bool
    modelUid: str
    uid: str
    views: dict


class IFeature(Protocol):
    attributes: dict
    cssSelector: str
    errors: list['ModelValidationError']
    isNew: bool
    layerName: str
    model: 'IModel'
    props: 'FeatureProps'
    record: 'FeatureRecord'
    views: dict
    createWithFeatures: list['IFeature']
    insertedPrimaryKey: str

    def get(self, name: str, default=None) -> Any: ...

    def has(self, name: str) -> bool: ...

    def set(self, name: str, value: Any) -> 'IFeature': ...

    def raw(self, name: str) -> Any: ...

    def render_views(self, templates: list['ITemplate'], **kwargs) -> 'IFeature': ...

    def shape(self) -> Optional['IShape']: ...

    def to_geojson(self, user: 'IUser') -> dict: ...

    def to_svg(self, view: 'MapView', label: Optional[str] = None, style: Optional['IStyle'] = None) -> list[IXmlElement]: ...

    def transform_to(self, crs: 'ICrs') -> 'IFeature': ...

    def uid(self) -> FeatureUid: ...


# ----------------------------------------------------------------------------------------------------------------------
# models

class ModelValidationError(Data):
    fieldName: str
    message: str


class ModelOperation(Enum):
    read = 'read'
    create = 'create'
    update = 'update'
    delete = 'delete'


class ModelReadMode(Enum):
    render = 'render'
    search = 'search'
    list = 'list'
    form = 'form'


class ModelDbSelect(Data):
    columns: list['sqlalchemy.Column']
    geometryWhere: list
    keywordWhere: list
    where: list
    order: list


class ModelContext(Data):
    op: ModelOperation
    readMode: ModelReadMode
    user: 'IUser'
    project: 'IProject'
    relDepth: int = 0
    maxDepth: int = 0
    search: 'SearchQuery'
    dbSelect: ModelDbSelect
    dbConnection: 'sqlalchemy.Connection'


EmptyValue = object()
ErrorValue = object()


class IModelWidget(INode, Protocol):
    pass


class IModelValidator(INode, Protocol):
    message: str
    ops: set[ModelOperation]

    def validate(self, field: 'IModelField', feature: 'IFeature', mc: ModelContext) -> bool: ...


class IModelValue(INode, Protocol):
    isDefault: bool
    ops: set[ModelOperation]

    def compute(self, field: 'IModelField', feature: 'IFeature', mc: 'ModelContext') -> Any: ...


class IModelField(INode, Protocol):
    name: str
    type: str
    title: str

    attributeType: AttributeType

    widget: Optional['IModelWidget'] = None

    values: list['IModelValue']
    validators: list['IModelValidator']

    isPrimaryKey: bool
    isRequired: bool
    isUnique: bool
    isAuto: bool

    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    model: 'IModel'

    def before_select(self, mc: ModelContext): ...

    def after_select(self, features: list['IFeature'], mc: ModelContext): ...

    def before_create(self, feature: 'IFeature', mc: ModelContext): ...

    def after_create(self, feature: 'IFeature', mc: ModelContext): ...

    def before_create_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def after_create_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def before_update(self, feature: 'IFeature', mc: ModelContext): ...

    def after_update(self, feature: 'IFeature', mc: ModelContext): ...

    def before_delete(self, feature: 'IFeature', mc: ModelContext): ...

    def after_delete(self, feature: 'IFeature', mc: ModelContext): ...

    def do_init(self, feature: 'IFeature', mc: ModelContext): ...

    def do_init_related(self, to_feature: 'IFeature', mc: ModelContext): ...

    def do_validate(self, feature: 'IFeature', mc: ModelContext): ...

    def from_props(self, feature: 'IFeature', mc: ModelContext): ...

    def to_props(self, feature: 'IFeature', mc: ModelContext): ...

    def related_models(self) -> list['IModel']: ...

    def find_relatable_features(self, search: 'SearchQuery', mc: ModelContext) -> list['IFeature']: ...

    def raw_to_python(self, feature: 'IFeature', value, mc: ModelContext): ...

    def prop_to_python(self, feature: 'IFeature', value, mc: ModelContext): ...

    def python_to_raw(self, feature: 'IFeature', value, mc: ModelContext): ...

    def python_to_prop(self, feature: 'IFeature', value, mc: ModelContext): ...

    def describe(self) -> Optional[ColumnDescription]: ...


class IModel(INode, Protocol):
    defaultSort: list['SearchSort']
    fields: list['IModelField']
    geometryCrs: Optional['ICrs']
    geometryName: str
    geometryType: Optional[GeometryType]
    isEditable: bool
    loadingStrategy: 'FeatureLoadingStrategy'
    title: str
    uidName: str

    def find_features(self, search: 'SearchQuery', mc: ModelContext) -> list['IFeature']: ...

    def get_features(self, uids: Iterable[str | int], mc: ModelContext) -> list['IFeature']: ...

    def init_feature(self, feature: 'IFeature', mc: ModelContext): ...

    def create_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def update_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def delete_feature(self, feature: 'IFeature', mc: ModelContext) -> FeatureUid: ...

    def validate_feature(self, feature: 'IFeature', mc: ModelContext) -> bool: ...

    def feature_from_props(self, props: 'FeatureProps', mc: ModelContext) -> 'IFeature': ...

    def feature_to_props(self, feature: 'IFeature', mc: ModelContext) -> 'FeatureProps': ...

    def feature_to_view_props(self, feature: 'IFeature', mc: ModelContext) -> 'FeatureProps': ...

    def describe(self) -> Optional[DataSetDescription]: ...

    def field(self, name: str) -> Optional['IModelField']: ...

    def related_models(self) -> list['IModel']: ...


class IDatabaseModel(IModel, Protocol):
    provider: 'IDatabaseProvider'
    sqlFilter: str
    tableName: str

    def table(self) -> 'sqlalchemy.Table': ...

    def column(self, column_name: str) -> 'sqlalchemy.Column': ...

    def uid_column(self) -> 'sqlalchemy.Column': ...

    def connection(self) -> 'sqlalchemy.Connection': ...

    def execute(self, sql: 'sqlalchemy.Executable', mc: ModelContext, parameters=None) -> 'sqlalchemy.CursorResult': ...


class IModelManager(INode, Protocol):
    def get_model(self, uid: str, user: IUser = None, access: Access = None) -> Optional['IModel']: ...

    def locate_model(self, *objects, uid: str = None, user: IUser = None, access: Access = None) -> Optional['IModel']: ...

    def editable_models(self, project: 'IProject', user: 'IUser') -> list['IModel']: ...

    def default_model(self) -> 'IModel': ...


# ----------------------------------------------------------------------------------------------------------------------
# templates and rendering

class ImageFormat(Enum):
    """Image format"""

    png8 = 'png8'
    """png 8-bit"""
    png24 = 'png24'
    """png 24-bit"""


class IImage(IObject, Protocol):
    def size(self) -> Size: ...

    def add_box(self, color=None) -> 'IImage': ...

    def add_text(self, text: str, x=0, y=0, color=None) -> 'IImage': ...

    def compose(self, other: 'IImage', opacity=1) -> 'IImage': ...

    def crop(self, box) -> 'IImage': ...

    def paste(self, other: 'IImage', where=None) -> 'IImage': ...

    def resize(self, size: Size, **kwargs) -> 'IImage': ...

    def rotate(self, angle: int, **kwargs) -> 'IImage': ...

    def to_bytes(self, mime: Optional[str] = None) -> bytes: ...

    def to_path(self, path: str, mime: Optional[str] = None) -> str: ...


class MapView(Data):
    bounds: Bounds
    center: Point
    rotation: int
    scale: int
    mmSize: Size
    pxSize: Size
    dpi: int


class MapRenderInputPlaneType(Enum):
    features = 'features'
    image = 'image'
    imageLayer = 'imageLayer'
    svgLayer = 'svgLayer'
    svgSoup = 'svgSoup'


class MapRenderInputPlane(Data):
    type: MapRenderInputPlaneType
    features: list['IFeature']
    image: 'IImage'
    layer: 'ILayer'
    opacity: float
    soupPoints: list[Point]
    soupTags: list[Any]
    styles: list['IStyle']
    subLayers: list[str]


class MapRenderInput(Data):
    backgroundColor: int
    bbox: Extent
    center: Point
    crs: 'ICrs'
    dpi: int
    mapSize: MSize
    notify: Callable
    planes: list['MapRenderInputPlane']
    project: 'IProject'
    rotation: int
    scale: int
    user: 'IUser'
    visibleLayers: Optional[list['ILayer']]


class MapRenderOutputPlaneType(Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


class MapRenderOutputPlane(Data):
    type: MapRenderOutputPlaneType
    path: str
    elements: list[IXmlElement]
    image: 'IImage'


class MapRenderOutput(Data):
    planes: list['MapRenderOutputPlane']
    view: MapView


class LayerRenderInputType(Enum):
    box = 'box'
    xyz = 'xyz'
    svg = 'svg'


class LayerRenderInput(Data):
    boxBuffer: int
    boxSize: int
    extraParams: dict
    project: 'IProject'
    style: 'IStyle'
    type: LayerRenderInputType
    user: 'IUser'
    view: MapView
    x: int
    y: int
    z: int


class LayerRenderOutput(Data):
    content: bytes
    tags: list[IXmlElement]


class TemplateRenderInput(Data):
    args: dict
    crs: ICrs
    dpi: int
    localeUid: str
    maps: list[MapRenderInput]
    mimeOut: str
    notify: Callable
    project: 'IProject'
    user: 'IUser'


class TemplateQualityLevel(Data):
    name: str
    dpi: int


class ITemplate(INode, Protocol):
    mapSize: MSize
    mimeTypes: list[str]
    pageMargin: MExtent
    pageSize: MSize
    subject: str
    title: str

    def render(self, tri: TemplateRenderInput) -> ContentResponse: ...


class ITemplateManager(INode, Protocol):
    def locate_template(self, *objects, user: IUser = None, subject: str = None, mime: str = None) -> Optional['ITemplate']: ...

    def template_from_path(self, path: str) -> Optional['ITemplate']: ...


class IPrinter(INode, Protocol):
    title: str
    template: 'ITemplate'
    models: list['IModel']
    qualityLevels: list[TemplateQualityLevel]


class IPrinterManager(INode, Protocol):
    def printers_for_project(self, project: 'IProject', user: 'IUser') -> list['IPrinter']: ...


class JobState(Enum):
    init = 'init'
    """the job is being created"""
    open = 'open'
    """the job is just created and waiting for start"""
    running = 'running'
    """the job is running"""
    complete = 'complete'
    """the job has been completed successfully"""
    error = 'error'
    """there was an error"""
    cancel = 'cancel'
    """the job was cancelled"""


# ----------------------------------------------------------------------------------------------------------------------
# styles

class StyleValues(Data):
    fill: Color

    stroke: Color
    stroke_dasharray: list[int]
    stroke_dashoffset: int
    stroke_linecap: Literal['butt', 'round', 'square']
    stroke_linejoin: Literal['bevel', 'round', 'miter']
    stroke_miterlimit: int
    stroke_width: int

    marker: Literal['circle', 'square', 'arrow', 'cross']
    marker_fill: Color
    marker_size: int
    marker_stroke: Color
    marker_stroke_dasharray: list[int]
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
    label_padding: list[int]
    label_placement: Literal['start', 'end', 'middle']
    label_stroke: Color
    label_stroke_dasharray: list[int]
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
    cssSelector: str
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
    dayNamesLong: list[str]
    dayNamesShort: list[str]
    dayNamesNarrow: list[str]
    firstWeekDay: int
    language: str
    languageName: str
    monthNamesLong: list[str]
    monthNamesShort: list[str]
    monthNamesNarrow: list[str]
    numberDecimal: str
    numberGroup: str


# ----------------------------------------------------------------------------------------------------------------------
# metadata


class MetadataLink(Data):
    """Link metadata."""

    description: Optional[str]
    format: Optional[str]
    formatVersion: Optional[str]
    function: Optional[str]
    mimeType: Optional[str]
    scheme: Optional[str]
    title: Optional[str]
    type: Optional[str]
    url: Optional[Url]


class MetadataAccessConstraint(Data):
    title: Optional[str]
    type: Optional[str]


class MetadataLicense(Data):
    title: Optional[str]
    url: Optional[Url]


class MetadataAttribution(Data):
    title: Optional[str]
    url: Optional[Url]


class Metadata(Data):
    abstract: Optional[str]
    accessConstraints: Optional[list[MetadataAccessConstraint]]
    attribution: Optional[MetadataAttribution]
    authorityIdentifier: Optional[str]
    authorityName: Optional[str]
    authorityUrl: Optional[str]
    catalogCitationUid: Optional[str]
    catalogUid: Optional[str]
    fees: Optional[str]
    image: Optional[str]
    keywords: Optional[list[str]]
    language3: Optional[str]
    language: Optional[str]
    languageName: Optional[str]
    license: Optional[MetadataLicense]
    name: Optional[str]
    parentIdentifier: Optional[str]
    title: Optional[str]

    contactAddress: Optional[str]
    contactAddressType: Optional[str]
    contactArea: Optional[str]
    contactCity: Optional[str]
    contactCountry: Optional[str]
    contactEmail: Optional[str]
    contactFax: Optional[str]
    contactOrganization: Optional[str]
    contactPerson: Optional[str]
    contactPhone: Optional[str]
    contactPosition: Optional[str]
    contactProviderName: Optional[str]
    contactProviderSite: Optional[str]
    contactRole: Optional[str]
    contactUrl: Optional[str]
    contactZip: Optional[str]

    dateBegin: Optional[str]
    dateCreated: Optional[str]
    dateEnd: Optional[str]
    dateUpdated: Optional[str]

    inspireKeywords: Optional[list[str]]
    inspireMandatoryKeyword: Optional[str]
    inspireDegreeOfConformity: Optional[str]
    inspireResourceType: Optional[str]
    inspireSpatialDataServiceType: Optional[str]
    inspireSpatialScope: Optional[str]
    inspireSpatialScopeName: Optional[str]
    inspireTheme: Optional[str]
    inspireThemeName: Optional[str]
    inspireThemeNameEn: Optional[str]

    isoMaintenanceFrequencyCode: Optional[str]
    isoQualityConformanceExplanation: Optional[str]
    isoQualityConformanceQualityPass: Optional[bool]
    isoQualityConformanceSpecificationDate: Optional[str]
    isoQualityConformanceSpecificationTitle: Optional[str]
    isoQualityLineageSource: Optional[str]
    isoQualityLineageSourceScale: Optional[int]
    isoQualityLineageStatement: Optional[str]
    isoRestrictionCode: Optional[str]
    isoServiceFunction: Optional[str]
    isoScope: Optional[str]
    isoScopeName: Optional[str]
    isoSpatialRepresentationType: Optional[str]
    isoTopicCategories: Optional[list[str]]
    isoSpatialResolution: Optional[str]

    metaLinks: Optional[list[MetadataLink]]
    serviceMetaLink: Optional[MetadataLink]
    extraLinks: Optional[list[MetadataLink]]


# ----------------------------------------------------------------------------------------------------------------------
# search


class SearchSort(Data):
    fieldName: str
    reverse: bool


class SearchOgcFilter(Data):
    name: str
    operator: str
    shape: 'IShape'
    subFilters: list['SearchOgcFilter']
    value: str


class SearchQuery(Data):
    access: Access
    all: bool
    bounds: Bounds
    extraColumns: list
    extraParams: dict
    extraWhere: list
    keyword: str
    layers: list['ILayer']
    limit: int
    ogcFilter: SearchOgcFilter
    project: 'IProject'
    relDepth: int
    resolution: float
    shape: 'IShape'
    sort: list[SearchSort]
    tolerance: 'Measurement'
    uids: list[str]


class SearchResult(Data):
    feature: 'IFeature'
    layer: 'ILayer'
    finder: 'IFinder'


class TextSearchType(Enum):
    exact = 'exact'
    """match the whole string"""
    begin = 'begin'
    """match the beginning of the string"""
    end = 'end'
    """match the end of the string"""
    any = 'any'
    """match any substring"""
    like = 'like'
    """use the percent sign as a placeholder"""


class TextSearchOptions(Data):
    type: TextSearchType
    """type of the search"""
    minLength: int = 0
    """minimal pattern length"""
    caseSensitive: bool = False
    """use the case sensitive search"""


class SortOptions(Data):
    fieldName: str
    reverse: bool = False


class ISearchManager(INode, Protocol):
    def run_search(self, search: 'SearchQuery', user: IUser) -> list['SearchResult']: ...


class IFinder(INode, Protocol):
    supportsFilterSearch: bool = False
    supportsGeometrySearch: bool = False
    supportsKeywordSearch: bool = False

    withFilter: bool
    withGeometry: bool
    withKeyword: bool

    templates: list['ITemplate']
    models: list['IModel']

    provider: 'IProvider'
    sourceLayers: list['SourceLayer']

    tolerance: 'Measurement'

    def run(self, search: SearchQuery, user: IUser, layer: Optional['ILayer'] = None) -> list['IFeature']: ...

    def can_run(self, search: SearchQuery, user: IUser) -> bool: ...


# ----------------------------------------------------------------------------------------------------------------------
# maps and layers


class IMap(INode, Protocol):
    rootLayer: 'ILayer'

    bounds: Bounds
    center: Point
    coordinatePrecision: int
    initResolution: float
    resolutions: list[float]
    title: str
    wgsExtent: Extent


class LegendRenderOutput(Data):
    html: str
    image: 'IImage'
    image_path: str
    size: Size
    mime: str


class ILegend(INode, Protocol):
    def render(self, args: Optional[dict] = None) -> Optional[LegendRenderOutput]: ...


class LayerDisplayMode(Enum):
    """Layer display mode"""

    box = 'box'
    """display a layer as one big image (WMS-alike)"""
    tile = 'tile'
    """display a layer in a tile grid"""
    client = 'client'
    """draw a layer in the client"""


class LayerClientOptions(Data):
    """Client options for a layer"""

    expanded: bool
    """the layer is expanded in the list view"""
    unlisted: bool
    """the layer is hidden in the list view"""
    selected: bool
    """the layer is initially selected"""
    hidden: bool
    """the layer is initially hidden"""
    unfolded: bool
    """the layer is not listed, but its children are"""
    exclusive: bool
    """only one of this layer's children is visible at a time"""


class TileGrid(Data):
    uid: str
    bounds: Bounds
    origin: Origin
    resolutions: list[float]
    tileSize: int


class LayerCache(Data):
    maxAge: int
    maxLevel: int
    requestBuffer: int
    requestTiles: int


class FeatureLoadingStrategy(Enum):
    """Loading strategy for features."""
    all = 'all'
    """load all features"""
    bbox = 'bbox'
    """load only features in the current map extent"""
    lazy = 'lazy'
    """load features on demand"""


class LayerOwsOptions(Data):
    enabled: bool
    layerName: str
    featureName: str
    xmlNamespace: 'XmlNamespace'
    geometryName: str


class ILayer(INode, Protocol):
    canRenderBox: bool
    canRenderSvg: bool
    canRenderXyz: bool

    isEnabledForOws: bool
    isGroup: bool
    isSearchable: bool

    hasLegend: bool

    bounds: Bounds
    wgsExtent: Extent
    mapCrs: 'ICrs'
    clientOptions: LayerClientOptions
    displayMode: LayerDisplayMode
    loadingStrategy: FeatureLoadingStrategy
    imageFormat: str
    opacity: float
    resolutions: list[float]
    title: str

    owsOptions: Optional['LayerOwsOptions']

    grid: Optional[TileGrid]
    cache: Optional[LayerCache]

    metadata: 'Metadata'
    legend: Optional['ILegend']
    legendUrl: str

    finders: list['IFinder']
    templates: list['ITemplate']
    models: list['IModel']

    layers: list['ILayer']

    provider: 'IProvider'
    sourceLayers: list['SourceLayer']

    def ancestors(self) -> list['ILayer']: ...

    def descendants(self) -> list['ILayer']: ...

    def render(self, lri: LayerRenderInput) -> Optional['LayerRenderOutput']: ...

    def get_features_for_view(self, search: SearchQuery, user: 'IUser', view_names: Optional[list[str]] = None) -> list['IFeature']: ...

    def render_legend(self, args: Optional[dict] = None) -> Optional['LegendRenderOutput']: ...

    def url_path(self, kind: str) -> str: ...


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
    allowedParameters: dict[str, list[str]]
    constraints: dict[str, list[str]]
    formats: list[str]
    params: dict[str, str]
    postUrl: Url
    preferredFormat: str
    url: Url
    verb: OwsVerb


class OwsCapabilities(Data):
    metadata: 'Metadata'
    operations: list['OwsOperation']
    sourceLayers: list['SourceLayer']
    tileMatrixSets: list['TileMatrixSet']
    version: str


class IOwsService(INode, Protocol):
    isRasterService: bool
    isVectorService: bool

    metadata: 'Metadata'
    name: str
    protocol: OwsProtocol
    supportedBounds: list[Bounds]
    supportedVersions: list[str]
    supportedOperations: list['OwsOperation']
    templates: list['ITemplate']
    updateSequence: str
    version: str
    withInspireMeta: bool
    withStrictParams: bool

    def handle_request(self, req: 'IWebRequester') -> ContentResponse: ...


class IOwsProvider(INode, Protocol):
    alwaysXY: bool
    forceCrs: 'ICrs'
    maxRequests: int
    metadata: 'Metadata'
    operations: list[OwsOperation]
    protocol: OwsProtocol
    sourceLayers: list['SourceLayer']
    url: Url
    version: str

    def get_operation(self, verb: OwsVerb, method: Optional[RequestMethod] = None) -> Optional[OwsOperation]: ...

    def get_features(self, args: SearchQuery, source_layers: list[SourceLayer]) -> list[FeatureRecord]: ...


class IOwsModel(IModel, Protocol):
    provider: 'IOwsProvider'
    sourceLayers: list['SourceLayer']

    def get_operation(self, verb: OwsVerb, method: Optional[RequestMethod] = None) -> Optional[OwsOperation]: ...


# ----------------------------------------------------------------------------------------------------------------------


# CLI

class CliParams(Data):
    """CLI params"""
    pass


# ----------------------------------------------------------------------------------------------------------------------
# actions and apis


class IActionManager(INode, Protocol):
    def actions_for_project(self, project: 'IProject', user: IUser) -> list['IAction']: ...

    def locate_action(self, project: Optional['IProject'], ext_type: str, user: IUser) -> Optional['IAction']: ...

    def prepare_action(
            self,
            command_category: str,
            command_name: str,
            params: dict,
            user: 'IUser',
            read_options=None,
    ): tuple[Callable, Request]: ...


class IAction(INode, Protocol):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# projects


class IClient(INode, Protocol):
    options: dict
    elements: list


class IProject(INode, Protocol):
    assetsRoot: Optional['WebDocumentRoot']
    client: 'IClient'

    localeUids: list[str]
    map: 'IMap'
    metadata: 'Metadata'

    actions: list['IAction']
    finders: list['IFinder']
    models: list['IModel']
    printers: list['IPrinter']
    templates: list['ITemplate']
    owsServices: list['IOwsService']


# ----------------------------------------------------------------------------------------------------------------------
# application

class IMonitor(INode, Protocol):
    def add_directory(self, path: str, pattern: Regex): ...

    def add_file(self, path: str): ...

    def start(self): ...


WebMiddlewareHandler: TypeAlias = Callable[['IWebRequester', Callable], 'IWebResponder']


class IMiddleware(Protocol):
    def enter_middleware(self, req: 'IWebRequester') -> Optional['IWebResponder']: ...

    def exit_middleware(self, req: 'IWebRequester', res: 'IWebResponder'): ...


class IApplication(INode, Protocol):
    client: 'IClient'
    localeUids: list[str]
    metadata: 'Metadata'
    monitor: 'IMonitor'
    qgisVersion: str
    version: str
    versionString: str
    defaultPrinter: 'IPrinter'

    actionMgr: 'IActionManager'
    authMgr: 'IAuthManager'
    databaseMgr: 'IDatabaseManager'
    modelMgr: 'IModelManager'
    printerMgr: 'IPrinterManager'
    searchMgr: 'ISearchManager'
    storageMgr: 'IStorageManager'
    templateMgr: 'ITemplateManager'
    webMgr: 'IWebManager'

    actions: list['IAction']
    finders: list['IFinder']
    templates: list['ITemplate']
    printers: list['IPrinter']
    models: list['IModel']
    owsServices: list['IOwsService']

    def register_middleware(self, name: str, obj: IMiddleware, depends_on=Optional[list[str]]): ...

    def middleware_objects(self) -> list[tuple[str, IMiddleware]]: ...

    def projects_for_user(self, user: 'IUser') -> list['IProject']: ...

    def project(self, uid: str) -> Optional['IProject']: ...

    def helper(self, ext_type: str) -> Optional['INode']: ...

    def developer_option(self, name: str): ...

    def require_helper(self, ext_type: str): ...

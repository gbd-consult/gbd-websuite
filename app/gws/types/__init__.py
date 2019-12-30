"""Common types and data structures."""    

#
# automatically generated from
#
# - types/__init__.in.py    
# - includes in types/t
# - class stubs generated from #:export comments
# 
    
# noinspection PyUnresolvedReferences
from typing import Any, Dict, List, Optional, Tuple, Union, cast
# noinspection PyUnresolvedReferences
from .data import Data, Config, Props

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


# noinspection PyPep8Naming
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

class CorsOptions(Data):
    allow_credentials: bool = None
    allow_headers: Optional[List[str]] = None
    allow_origin: str = None

class DocumentRoot(Data):
    allow_mime: Optional[List[str]] = None
    deny_mime: Optional[List[str]] = None
    dir: 'DirPath' = None

class FeatureConvertor:
    data_model: 'IModel' = None
    feature_format: 'IFormat' = None

class FeatureProps(Data):
    attributes: Optional[List[Attribute]] = None
    elements: Optional[dict] = None
    layerUid: Optional[str] = None
    shape: Optional['ShapeProps'] = None
    style: Optional['StyleProps'] = None
    uid: Optional[str] = None

class IBaseRequest:
    cookies: dict = None
    data: Optional[bytes] = None
    environ: dict = None
    headers: dict = None
    input_struct_type: int = None
    kparams: dict = None
    method: str = None
    output_struct_type: int = None
    params: dict = None
    root: 'IRootObject' = None
    site: 'IWebSite' = None
    text_data: Optional[str] = None
    def env(self, key: str, default: str = None) -> str: pass
    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> 'IResponse': pass
    def kparam(self, key: str, default: str = None) -> str: pass
    def param(self, key: str, default: str = None) -> str: pass
    def response(self, content: str, mimetype: str, status: int = 200) -> 'IResponse': pass
    def struct_response(self, data: 'Response', status: int = 200) -> 'IResponse': pass
    def url_for(self, url: 'Url') -> 'Url': pass

class IFeature:
    attributes: List[Attribute] = None
    convertor: 'FeatureConvertor' = None
    elements: dict = None
    layer: 'ILayer' = None
    props: 'FeatureProps' = None
    shape: 'IShape' = None
    style: 'IStyle' = None
    uid: str = None
    def convert(self, target_crs: 'Crs' = None, convertor: 'FeatureConvertor' = None) -> 'IFeature': pass
    def to_geojson(self) -> dict: pass
    def to_svg(self, rv: 'RenderView', style: 'IStyle' = None) -> str: pass
    def transform(self, to_crs) -> 'IFeature': pass

class IObject:
    auto_uid: str = None
    children: list = None
    config: Config = None
    parent: 'IObject' = None
    props: Props = None
    root: 'IRootObject' = None
    uid: str = None
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

class IResponse:
    pass

class IRole:
    def can_use(self, obj, parent=None): pass

class IShape:
    bounds: 'Extent' = None
    centroid: 'IShape' = None
    crs: 'Crs' = None
    props: 'ShapeProps' = None
    type: str = None
    wkb: str = None
    wkb_hex: str = None
    wkt: str = None
    x: float = None
    y: float = None
    def intersects(self, shape: 'IShape') -> bool: pass
    def tolerance_buffer(self, tolerance, resolution=None) -> 'IShape': pass
    def transform(self, to_crs) -> 'IShape': pass

class IStyle:
    content: dict = None
    props: 'StyleProps' = None
    text: str = None
    type: str = None

class IUser:
    attributes: dict = None
    display_name: str = None
    full_uid: str = None
    is_guest: bool = None
    props: 'UserProps' = None
    provider: 'IAuthProvider' = None
    roles: List[str] = None
    uid: str = None
    def attribute(self, key: str, default: str = '') -> str: pass
    def can_use(self, obj: 'IObject', parent: 'IObject' = None) -> bool: pass
    def has_role(self, role: str) -> bool: pass
    def init_from_cache(self, provider, uid, roles, attributes) -> 'IUser': pass
    def init_from_source(self, provider, uid, roles=None, attributes=None) -> 'IUser': pass

class MetaContact(Data):
    address: str = None
    area: str = None
    city: str = None
    country: str = None
    email: str = None
    fax: str = None
    organization: str = None
    person: str = None
    phone: str = None
    position: str = None
    url: 'Url' = None
    zip: str = None

class MetaData(Data):
    abstract: str = None
    accessConstraints: str = None
    attribution: str = None
    contact: Optional['MetaContact'] = None
    fees: str = None
    image: 'Url' = None
    images: dict = None
    inspire: dict = None
    iso: dict = None
    keywords: List[str] = None
    language: str = None
    links: List['MetaLink'] = None
    modDate: 'Date' = None
    name: str = None
    pubDate: 'Date' = None
    serviceUrl: 'Url' = None
    title: str = None
    url: 'Url' = None

class MetaLink(Data):
    function: str = None
    scheme: str = None
    url: 'Url' = None

class ModelProps(Props):
    rules: List['ModelRule'] = None

class ModelRule(Data):
    expression: str = None
    format: 'FormatStr' = None
    name: str = None
    source: str = None
    title: str = None
    type: 'AttributeType' = None
    value: Optional[str]  #: constant value = None

class OwsOperation:
    formats: List[str] = None
    get_url: 'Url' = None
    name: str = None
    parameters: dict = None
    post_url: 'Url' = None

class RenderInput(Data):
    background_color: int = None
    items: List['RenderInputItem'] = None
    view: 'RenderView' = None

class RenderInputItem(Data):
    dpi: int = None
    features: List['IFeature'] = None
    fragment: 'SvgFragment' = None
    layer: 'ILayer' = None
    opacity: float = None
    print_as_vector: bool = None
    style: 'IStyle' = None
    sub_layers: List[str] = None
    type: str = None

class RenderInputItemType:
    bbox_layer: str = None
    features: str = None
    fragment: str = None
    image: str = None
    svg_layer: str = None

class RenderOutput(Data):
    items: List['RenderOutputItem'] = None
    view: 'RenderView' = None

class RenderOutputItem(Data):
    elements: List[str] = None
    path: str = None
    type: str = None

class RenderOutputItemType:
    image: str = None
    path: str = None
    svg: str = None

class RenderView(Data):
    bbox: 'Extent' = None
    center: 'Point' = None
    dpi: int = None
    rotation: int = None
    scale: int = None
    size_mm: 'Size' = None
    size_px: 'Size' = None

class RewriteRule(Data):
    match: 'Regex'  #: expression to match the url against = None
    options: Optional[dict]  #: additional options = None
    target: str  #: target url with placeholders = None

class SearchArgs(Data):
    axis: str = None
    bbox: 'Extent' = None
    count: int = None
    crs: 'Crs' = None
    feature_format: 'IFormat' = None
    keyword: Optional[str] = None
    layers: List['ILayer'] = None
    limit: int = None
    params: dict = None
    point: 'Point' = None
    project: 'IProject' = None
    resolution: float = None
    shapes: List['IShape'] = None
    tolerance: int = None

class SelectArgs(Data):
    extraWhere: Optional[str] = None
    keyword: Optional[str] = None
    limit: Optional[int] = None
    shape: Optional['IShape'] = None
    sort: Optional[str] = None
    table: 'SqlTable' = None
    tolerance: Optional[float] = None
    uids: Optional[List[str]] = None

class ShapeProps(Props):
    crs: str = None
    geometry: dict = None

class SourceLayer(Data):
    a_level: int = None
    a_path: str = None
    a_uid: str = None
    data_source: dict = None
    extents: Dict['Crs', 'Extent'] = None
    is_expanded: bool = None
    is_group: bool = None
    is_image: bool = None
    is_queryable: bool = None
    is_visible: bool = None
    layers: List['SourceLayer'] = None
    legend: str = None
    meta: 'MetaData' = None
    name: str = None
    opacity: int = None
    resource_urls: dict = None
    scale_range: List[float] = None
    styles: List['SourceStyle'] = None
    supported_crs: List['Crs'] = None
    title: str = None

class SourceStyle(Data):
    is_default: bool = None
    legend: 'Url' = None
    meta: 'MetaData' = None

class SqlTable(Data):
    geometry_column: str = None
    geometry_crs: 'Crs' = None
    geometry_type: str = None
    key_column: str = None
    name: str = None
    search_column: str = None

class SqlTableColumn(Data):
    crs: 'Crs' = None
    is_geometry: bool = None
    is_key: bool = None
    name: str = None
    native_type: str = None
    type: 'AttributeType' = None

class StorageEntry(Data):
    category: str = None
    name: str = None

class StyleProps(Props):
    content: Optional[dict] = None
    text: Optional[str] = None
    type: str = None

class SvgFragment:
    points: List['Point'] = None
    svg: str = None

class TemplateOutput(Data):
    content: str = None
    mime: str = None
    path: str = None

class TemplateProps(Props):
    dataModel: 'ModelProps' = None
    mapHeight: int = None
    mapWidth: int = None
    qualityLevels: List['TemplateQualityLevel'] = None
    title: str = None
    uid: str = None

class TemplateQualityLevel(Data):
    dpi: int  #: dpi value = None
    name: str = None

class UserProps(Data):
    displayName: str = None

class IApi(IObject):
    actions: dict = None

class IApplication(IObject):
    api: 'IApi' = None
    client: 'IClient' = None
    qgis_version: str = None
    storage: 'IStorage' = None
    web_sites: List['IWebSite'] = None
    def find_action(self, action_type, project_uid=None): pass

class IAuthProvider(IObject):
    def authenticate(self, login: str, password: str, **kwargs) -> 'IUser': pass
    def get_user(self, user_uid: str) -> 'IUser': pass
    def marshal_user(self, u: 'IUser') -> str: pass
    def unmarshal_user(self, user_uid: str, json: str) -> 'IUser': pass

class IClient(IObject):
    pass

class IDbProvider(IObject):
    pass

class IFormat(IObject):
    templates: dict = None
    def apply(self, context: dict) -> dict: pass

class ILayer(IObject):
    can_render_bbox: bool = None
    can_render_svg: bool = None
    can_render_xyz: bool = None
    crs: str = None
    data_model: 'IModel' = None
    description: str = None
    description_template: 'ITemplate' = None
    display: str = None
    edit_data_model: 'IModel' = None
    edit_style: 'IStyle' = None
    extent: list = None
    feature_format: 'IFormat' = None
    has_cache: bool = None
    has_legend: bool = None
    has_search: bool = None
    image_format: str = None
    is_public: bool = None
    layers: list = None
    legend_url: str = None
    map: 'IMap' = None
    meta: 'MetaData' = None
    opacity: int = None
    ows_name: str = None
    resolutions: list = None
    services: list = None
    style: 'IStyle' = None
    title: str = None
    def edit_access(self, user): pass
    def edit_operation(self, operation: str, feature_props: List['FeatureProps']) -> List['IFeature']: pass
    def get_features(self, bbox: 'Extent', limit: int = 0) -> List['IFeature']: pass
    def mapproxy_config(self, mc): pass
    def ows_enabled(self, service): pass
    def render_bbox(self, rv: 'RenderView', client_params=None): pass
    def render_legend(self): pass
    def render_svg(self, rv: 'RenderView', style=None): pass
    def render_xyz(self, x, y, z): pass
    def use_meta(self, meta): pass

class IMap(IObject):
    center: list = None
    coordinate_precision: int = None
    crs: str = None
    extent: list = None
    init_resolution: int = None
    layers: List['ILayer'] = None
    resolutions: list = None

class IModel(IObject):
    rules: List['ModelRule'] = None
    def apply(self, atts: List[Attribute]) -> List[Attribute]: pass
    def apply_to_dict(self, d: dict) -> List[Attribute]: pass

class IOwsProvider(IObject):
    meta: 'MetaData' = None
    operations: List['OwsOperation'] = None
    source_layers: List['SourceLayer'] = None
    supported_crs: List['Crs'] = None
    type: str = None
    url: 'Url' = None
    version: str = None
    def find_features(self, args: 'SearchArgs') -> List['IFeature']: pass
    def operation(self, name: str) -> 'OwsOperation': pass

class IOwsService(IObject):
    feature_namespace: str = None
    local_namespaces: dict = None
    name: str = None
    templates: dict = None
    type: str = None
    use_inspire_data: bool = None
    use_inspire_meta: bool = None
    version: str = None
    def configure_inspire_templates(self): pass
    def configure_template(self, name, path, type='xml'): pass
    def error_response(self, status): pass
    def handle(self, req) -> 'HttpResponse': pass
    def is_layer_enabled(self, layer): pass

class IPrinter(IObject):
    templates: List['ITemplate'] = None

class IProject(IObject):
    api: 'IApi' = None
    assets_root: 'DocumentRoot' = None
    client: 'IClient' = None
    description_template: 'ITemplate' = None
    locales: list = None
    map: 'IMap' = None
    meta: 'MetaData' = None
    overview_map: 'IMap' = None
    printer: 'IPrinter' = None
    title: str = None

class IRequest(IBaseRequest):
    user: 'IUser' = None
    def acquire(self, klass: str, uid: str) -> 'IObject': pass
    def auth_begin(self): pass
    def auth_commit(self, res): pass
    def login(self, username: str, password: str): pass
    def logout(self): pass
    def require(self, klass: str, uid: str) -> 'IObject': pass
    def require_project(self, uid: str) -> 'IProject': pass

class ISearchProvider(IObject):
    geometry_required: bool = None
    keyword_required: bool = None
    def can_run(self, args: 'SearchArgs'): pass
    def context_shape(self, args: 'SearchArgs'): pass
    def run(self, layer: 'ILayer', args: 'SearchArgs') -> List['IFeature']: pass

class IStorage(IObject):
    def can_read(self, r, user: 'IUser') -> bool: pass
    def can_write(self, r, user: 'IUser') -> bool: pass
    def dir(self, user: 'IUser') -> List['StorageEntry']: pass
    def read(self, entry: 'StorageEntry', user: 'IUser') -> dict: pass
    def write(self, entry: 'StorageEntry', user: 'IUser', data: dict) -> bool: pass

class ITemplate(IObject):
    data_model: 'IModel' = None
    map_size: 'Size' = None
    page_size: 'Size' = None
    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str: pass
    def dpi_for_quality(self, quality): pass
    def normalize_user_data(self, d: dict) -> List[Attribute]: pass
    def render(self, context: dict, render_output: 'RenderOutput' = None, out_path: str = None, format: str = None) -> 'TemplateOutput': pass

class IWebSite(IObject):
    assets_root: 'DocumentRoot' = None
    cors: 'CorsOptions' = None
    error_page: 'ITemplate' = None
    host: str = None
    reversed_rewrite_rules: List['RewriteRule'] = None
    rewrite_rules: List['RewriteRule'] = None
    ssl: bool = None
    static_root: 'DocumentRoot' = None
    def url_for(self, req, url): pass

class RootBase(IObject):
    all_objects: list = None
    all_types: dict = None
    shared_objects: dict = None
    def create(self, klass, cfg=None): pass

class IRootObject(RootBase):
    application: 'IApplication' = None
    def configure(self): pass
    def validate_action(self, category, cmd, payload): pass

class ISqlProvider(IDbProvider):
    def describe(self, table: 'SqlTable') -> Dict[str, 'SqlTableColumn']: pass
    def edit_operation(self, operation: str, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: pass
    def select(self, args: 'SelectArgs', extra_connect_params: dict = None) -> List['IFeature']: pass

class IVectorLayer(ILayer):
    def connect_feature(self, feature: 'IFeature') -> 'IFeature': pass
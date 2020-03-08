from typing import Any, Dict, List, Optional, Tuple, Union, cast


# NB: we cannot use the standard Enum, because after "class Color(Enum): RED = 1"
# the value of Color.RED is like {'_value_': 1, '_name_': 'RED', '__objclass__': etc}
# and we need it to be 1, literally (that's what we'll get from the client)

class Enum:
    pass


#:alias An array of 4 elements representing extent coordinates [minx, miny, maxx, maxy]
Extent = Tuple[float, float, float, float]

#:alias Point coordinates [x, y]
Point = Tuple[float, float]

#:alias Size [width, height]
Size = Tuple[float, float]

#:alias A value with a unit
Measurement = Tuple[float, str]


class Axis(Enum):
    """Axis orientation."""
    xy = 'xy'
    yx = 'yx'


#:alias Verbatim literal type
Literal = str

#:alias Valid readable file path on the server
FilePath = str

#:alias Valid readable directory path on the server
DirPath = str

#:alias String like "1w 2d 3h 4m 5s" or a number of seconds
Duration = str

#:alias CSS color name
Color = str

#:alias Regular expression, as used in Python
Regex = str

#:alias String with {attribute} placeholders
FormatStr = str

#:alias CRS code like "EPSG:3857
Crs = str

#:alias ISO date like "2019-01-30"
Date = str

#:alias Http or https URL
Url = str


# dummy classes to support extension typing

class ext:
    class action:
        class Config:
            pass

        class Props:
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

    class helper:
        class Config:
            pass

    class ows:
        class provider:
            class Config:
                pass

        class service:
            class Config:
                pass


# basic data type

class Data:
    """Basic data object."""

    def __init__(self, *args, **kwargs):
        self._extend(args, kwargs)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def as_dict(self):
        return vars(self)

    def __repr__(self):
        return repr(vars(self))

    def _extend(self, args, kwargs):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
            elif hasattr(a, 'as_dict'):
                d.update(a.as_dict())
        d.update(kwargs)
        vars(self).update(d)


# configuration primitives

class Config(Data):
    """Configuration base type"""

    uid: str = ''  #: unique ID


class WithType(Config):
    type: str  #: object type


class AccessType(Enum):
    allow = 'allow'
    deny = 'deny'


class AccessRuleConfig(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithAccess(Config):
    access: Optional[List[AccessRuleConfig]]  #: access rights


class WithTypeAndAccess(Config):
    type: str  #: object type
    access: Optional[List[AccessRuleConfig]]  #: access rights


# attributes

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
    type: AttributeType = 'str'
    value: Any = None
    editable: bool = True


# request params and responses

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


# props baseclass


class Props(Data):
    """Properties base type"""
    pass


class Bounds(Data):
    crs: 'Crs' = None
    extent: 'Extent' = None


class CorsOptions(Data):
    allow_credentials: bool = None
    allow_headers: Optional[List[str]] = None
    allow_origin: str = None


class DocumentRoot(Data):
    allow_mime: Optional[List[str]] = None
    deny_mime: Optional[List[str]] = None
    dir: 'DirPath' = None


class FeatureConverter:
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
    method: str = None
    output_struct_type: int = None
    params: dict = None
    root: 'IRootObject' = None
    site: 'IWebSite' = None
    text_data: Optional[str] = None
    def env(self, key: str, default: str = None) -> str: pass
    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> 'IResponse': pass
    def has_param(self, key: str) -> bool: pass
    def param(self, key: str, default: str = None) -> str: pass
    def parse_params(self): pass
    def response(self, content: str, mimetype: str, status: int = 200) -> 'IResponse': pass
    def struct_response(self, data: 'Response', status: int = 200) -> 'IResponse': pass
    def url_for(self, url: 'Url') -> 'Url': pass


class IFeature:
    attributes: List[Attribute] = None
    category: str = None
    converter: Optional['FeatureConverter'] = None
    elements: dict = None
    full_uid: str = None
    layer: Optional['ILayer'] = None
    minimal_props: 'FeatureProps' = None
    props: 'FeatureProps' = None
    shape: Optional['IShape'] = None
    style: Optional['IStyle'] = None
    template_context: dict = None
    uid: str = None
    def apply_converter(self, converter: 'FeatureConverter' = None) -> 'IFeature': pass
    def apply_data_model(self, model: 'IModel') -> 'IFeature': pass
    def apply_format(self, fmt: 'IFormat', extra_context: dict = None) -> 'IFeature': pass
    def attr(self, name: str): pass
    def to_geojson(self) -> dict: pass
    def to_svg(self, rv: 'RenderView', style: 'IStyle' = None) -> str: pass
    def transform_to(self, crs) -> 'IFeature': pass


class IObject:
    children: list = None
    config: Config = None
    parent: 'IObject' = None
    props: Props = None
    root: 'IRootObject' = None
    uid: str = None
    def add_child(self, klass, cfg): pass
    def append_child(self, obj): pass
    def configure(self): pass
    def create_object(self, klass, cfg, parent=None): pass
    def create_shared_object(self, klass, uid, cfg): pass
    def create_unbound_object(self, klass, cfg): pass
    def find(self, klass, uid) -> 'IObject': pass
    def find_all(self, klass=None) -> List['IObject']: pass
    def find_first(self, klass) -> 'IObject': pass
    def get_children(self, klass) -> List['IObject']: pass
    def get_closest(self, klass) -> 'IObject': pass
    def initialize(self, cfg): pass
    def is_a(self, klass): pass
    def post_configure(self): pass
    def post_initialize(self): pass
    def props_for(self, user) -> Optional[dict]: pass
    def set_uid(self, uid): pass
    def var(self, key, default=None, parent=False): pass


class IResponse:
    pass


class IRole:
    def can_use(self, obj, parent=None): pass


class IShape:
    area: float = None
    bounds: 'Bounds' = None
    centroid: 'IShape' = None
    crs: str = None
    ewkb: bytes = None
    ewkb_hex: str = None
    ewkt: str = None
    extent: 'Extent' = None
    props: 'ShapeProps' = None
    srid: int = None
    type: 'GeometryType' = None
    wkb: bytes = None
    wkb_hex: str = None
    wkt: str = None
    x: float = None
    y: float = None
    def intersects(self, shape: 'IShape') -> bool: pass
    def to_type(self, new_type: 'GeometryType') -> 'IShape': pass
    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape': pass
    def transformed_to(self, to_crs, **kwargs) -> 'IShape': pass


class IStyle:
    props: 'StyleProps' = None
    text: str = None
    type: 'StyleType' = None
    values: 'StyleValues' = None


class IUser:
    attributes: dict = None
    display_name: str = None
    fid: str = None
    is_guest: bool = None
    props: 'UserProps' = None
    provider: 'IAuthProvider' = None
    roles: List[str] = None
    uid: str = None
    def attribute(self, key: str, default: str = '') -> str: pass
    def can_use(self, obj, parent=None) -> bool: pass
    def has_role(self, role: str) -> bool: pass
    def init_from_props(self, provider, uid, roles, attributes) -> 'IUser': pass
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
    url: str = None
    zip: str = None


class MetaData(Data):
    abstract: str = None
    accessConstraints: str = None
    attribution: str = None
    contact: 'MetaContact' = None
    fees: str = None
    image: 'Url' = None
    images: dict = None
    inspire: dict = None
    iso: dict = None
    keywords: List[str] = None
    language: str = None
    links: List['MetaLink'] = None
    modDate: str = None
    name: str = None
    pubDate: str = None
    serviceUrl: 'Url' = None
    title: str = None
    uid: str = None
    url: 'Url' = None


class MetaLink(Data):
    function: str = None
    scheme: str = None
    url: 'Url' = None


class ModelProps(Props):
    rules: List['ModelRule'] = None


class ModelRule(Data):
    editable: bool = None
    expression: str = None
    format: 'FormatStr' = None
    name: str = None
    source: str = None
    title: str = None
    type: 'AttributeType' = None
    value: Optional[str] = None


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


class RenderInputItemType(Enum):
    features = 'features'
    fragment = 'fragment'
    image = 'image'
    image_layer = 'image_layer'
    svg_layer = 'svg_layer'


class RenderOutput(Data):
    items: List['RenderOutputItem'] = None
    view: 'RenderView' = None


class RenderOutputItem(Data):
    elements: List[str] = None
    path: str = None
    type: str = None


class RenderOutputItemType(Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


class RenderView(Data):
    bounds: 'Bounds' = None
    center: 'Point' = None
    dpi: int = None
    rotation: int = None
    scale: int = None
    size_mm: 'Size' = None
    size_px: 'Size' = None


class RewriteRule(Data):
    match: 'Regex' = None
    options: Optional[dict] = None
    target: str = None


class SearchArgs(Data):
    axis: str = None
    bounds: 'Bounds' = None
    keyword: Optional[str] = None
    layers: List['ILayer'] = None
    limit: int = None
    params: dict = None
    project: 'IProject' = None
    resolution: float = None
    shapes: List['IShape'] = None
    source_layer_names: List[str] = None
    tolerance: 'Measurement' = None


class SelectArgs(Data):
    extra_where: Optional[str] = None
    keyword: Optional[str] = None
    limit: Optional[int] = None
    map_tolerance: Optional[float] = None
    shape: Optional['IShape'] = None
    sort: Optional[str] = None
    table: 'SqlTable' = None
    uids: Optional[List[str]] = None


class ShapeProps(Props):
    crs: str = None
    geometry: dict = None


class SourceLayer(Data):
    a_level: int = None
    a_path: str = None
    a_uid: str = None
    data_source: dict = None
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
    supported_bounds: List['Bounds'] = None
    supported_crs: List['Crs'] = None
    title: str = None


class SourceStyle(Data):
    is_default: bool = None
    legend: 'Url' = None
    meta: 'MetaData' = None


class SqlTable(Data):
    geometry_column: str = None
    geometry_crs: 'Crs' = None
    geometry_type: 'GeometryType' = None
    key_column: str = None
    name: str = None
    search_column: str = None


class SqlTableColumn(Data):
    crs: 'Crs' = None
    geom_type: 'GeometryType' = None
    is_geometry: bool = None
    is_key: bool = None
    name: str = None
    native_type: str = None
    type: 'AttributeType' = None


class StorageDirectory(Data):
    category: str = None
    entries: List['StorageEntry'] = None
    readable: bool = None
    writable: bool = None


class StorageElement(Data):
    data: dict = None
    entry: 'StorageEntry' = None


class StorageEntry(Data):
    category: str = None
    name: str = None


class StorageRecord(Data):
    category: str = None
    created: int = None
    data: str = None
    name: str = None
    updated: int = None
    user_fid: str = None


class StyleGeometryOption(Enum):
    all = 'all'
    none = 'none'


class StyleLabelAlign(Enum):
    center = 'center'
    left = 'left'
    right = 'right'


class StyleLabelFontStyle(Enum):
    italic = 'italic'
    normal = 'normal'


class StyleLabelFontWeight(Enum):
    bold = 'bold'
    normal = 'normal'


class StyleLabelOption(Enum):
    all = 'all'
    none = 'none'


class StyleLabelPlacement(Enum):
    end = 'end'
    middle = 'middle'
    start = 'start'


class StyleMarker(Enum):
    arrow = 'arrow'
    circle = 'circle'
    cross = 'cross'
    square = 'square'


class StyleProps(Props):
    text: Optional[str] = None
    type: 'StyleType' = None
    values: Optional['StyleValues'] = None


class StyleStrokeLineCap(Enum):
    butt = 'butt'
    round = 'round'
    square = 'square'


class StyleStrokeLineJoin(Enum):
    bevel = 'bevel'
    miter = 'miter'
    round = 'round'


class StyleType(Enum):
    css = 'css'
    cssSelector = 'cssSelector'


class StyleValues(Data):
    fill: Optional['Color'] = None
    icon: Optional[str] = None
    label_align: Optional['StyleLabelAlign'] = None
    label_background: Optional['Color'] = None
    label_fill: Optional['Color'] = None
    label_font_family: Optional[str] = None
    label_font_size: Optional[int] = None
    label_font_style: Optional['StyleLabelFontStyle'] = None
    label_font_weight: Optional['StyleLabelFontWeight'] = None
    label_line_height: Optional[int] = None
    label_max_scale: Optional[int] = None
    label_min_scale: Optional[int] = None
    label_offset_x: Optional[int] = None
    label_offset_y: Optional[int] = None
    label_padding: Optional[List[int]] = None
    label_placement: Optional['StyleLabelPlacement'] = None
    label_stroke: Optional['Color'] = None
    label_stroke_dasharray: Optional[List[int]] = None
    label_stroke_dashoffset: Optional[int] = None
    label_stroke_linecap: Optional['StyleStrokeLineCap'] = None
    label_stroke_linejoin: Optional['StyleStrokeLineJoin'] = None
    label_stroke_miterlimit: Optional[int] = None
    label_stroke_width: Optional[int] = None
    marker: Optional['StyleMarker'] = None
    marker_fill: Optional['Color'] = None
    marker_size: Optional[int] = None
    marker_stroke: Optional['Color'] = None
    marker_stroke_dasharray: Optional[List[int]] = None
    marker_stroke_dashoffset: Optional[int] = None
    marker_stroke_linecap: Optional['StyleStrokeLineCap'] = None
    marker_stroke_linejoin: Optional['StyleStrokeLineJoin'] = None
    marker_stroke_miterlimit: Optional[int] = None
    marker_stroke_width: Optional[int] = None
    offset_x: Optional[int] = None
    offset_y: Optional[int] = None
    point_size: Optional[int] = None
    stroke: Optional['Color'] = None
    stroke_dasharray: Optional[List[int]] = None
    stroke_dashoffset: Optional[int] = None
    stroke_linecap: Optional['StyleStrokeLineCap'] = None
    stroke_linejoin: Optional['StyleStrokeLineJoin'] = None
    stroke_miterlimit: Optional[int] = None
    stroke_width: Optional[int] = None
    with_geometry: Optional['StyleGeometryOption'] = None
    with_label: Optional['StyleLabelOption'] = None


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
    dpi: int = None
    name: str = None


class UserProps(Data):
    displayName: str = None


class IApi(IObject):
    actions: dict = None


class IApplication(IObject):
    api: 'IApi' = None
    client: 'IClient' = None
    qgis_version: str = None
    web_sites: List['IWebSite'] = None
    def find_action(self, action_type, project_uid=None): pass


class IAuthProvider(IObject):
    def authenticate(self, login: str, password: str, **kwargs) -> 'IUser': pass
    def get_user(self, user_uid: str) -> 'IUser': pass
    def user_from_dict(self, d: dict) -> 'IUser': pass
    def user_to_dict(self, u: 'IUser') -> dict: pass


class IClient(IObject):
    pass


class IDbProvider(IObject):
    pass


class IFormat(IObject):
    templates: dict = None
    def apply(self, context: dict) -> dict: pass


class ILayer(IObject):
    can_render_box: bool = None
    can_render_svg: bool = None
    can_render_xyz: bool = None
    data_model: 'IModel' = None
    default_search_provider: Optional['ISearchProvider'] = None
    description: str = None
    description_template: 'ITemplate' = None
    display: str = None
    edit_data_model: 'IModel' = None
    edit_options: Data = None
    edit_style: 'IStyle' = None
    extent: 'Extent' = None
    feature_format: 'IFormat' = None
    has_cache: bool = None
    has_legend: bool = None
    has_search: bool = None
    image_format: str = None
    is_editable: bool = None
    is_public: bool = None
    layers: list = None
    legend_url: str = None
    map: 'IMap' = None
    meta: 'MetaData' = None
    opacity: int = None
    own_bounds: Optional['Bounds'] = None
    ows_name: str = None
    ows_services_disabled: list = None
    ows_services_enabled: list = None
    resolutions: List[float] = None
    services: list = None
    style: 'IStyle' = None
    supports_wfs: bool = None
    supports_wms: bool = None
    title: str = None
    def edit_access(self, user): pass
    def edit_operation(self, operation: str, feature_props: List['FeatureProps']) -> List['IFeature']: pass
    def get_features(self, bounds: 'Bounds', limit: int = 0) -> List['IFeature']: pass
    def load_metadata(self, provider_meta=None): pass
    def mapproxy_config(self, mc): pass
    def ows_enabled(self, service: 'IOwsService') -> bool: pass
    def render_box(self, rv: 'RenderView', client_params=None): pass
    def render_legend(self): pass
    def render_svg(self, rv: 'RenderView', style: 'IStyle' = None): pass
    def render_xyz(self, x, y, z): pass


class IMap(IObject):
    bounds: 'Bounds' = None
    center: 'Point' = None
    coordinate_precision: int = None
    crs: 'Crs' = None
    extent: 'Extent' = None
    init_resolution: float = None
    layers: List['ILayer'] = None
    resolutions: List[float] = None


class IModel(IObject):
    attribute_names: List[str] = None
    geometry_crs: str = None
    geometry_type: 'GeometryType' = None
    is_identity: bool = None
    rules: List['ModelRule'] = None
    def apply(self, atts: List[Attribute]) -> List[Attribute]: pass
    def apply_to_dict(self, d: dict) -> List[Attribute]: pass


class IOwsProvider(IObject):
    invert_axis_crs: List[str] = None
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
    enabled: bool = None
    feature_namespace: str = None
    meta: 'MetaData' = None
    name: str = None
    type: str = None
    version: str = None
    def error_response(self, status) -> 'HttpResponse': pass
    def handle(self, req: 'IRequest') -> 'HttpResponse': pass


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
    def require_layer(self, uid: str) -> 'ILayer': pass
    def require_project(self, uid: str) -> 'IProject': pass


class ISearchProvider(IObject):
    data_model: 'IModel' = None
    feature_format: 'IFormat' = None
    tolerance: 'Measurement' = None
    with_geometry: str = None
    with_keyword: str = None
    def can_run(self, args: 'SearchArgs'): pass
    def context_shape(self, args: 'SearchArgs') -> 'IShape': pass
    def run(self, layer: 'ILayer', args: 'SearchArgs') -> List['IFeature']: pass


class ITemplate(IObject):
    data_model: 'IModel' = None
    map_size: 'Size' = None
    page_size: 'Size' = None
    path: str = None
    text: str = None
    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str: pass
    def dpi_for_quality(self, quality): pass
    def normalize_context(self, context: dict) -> dict: pass
    def render(self, context: dict, render_output: 'RenderOutput' = None, out_path: str = None, format: str = None) -> 'TemplateOutput': pass


class IWebSite(IObject):
    assets_root: 'DocumentRoot' = None
    cors: 'CorsOptions' = None
    error_page: 'ITemplate' = None
    host: str = None
    reversed_host: str = None
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
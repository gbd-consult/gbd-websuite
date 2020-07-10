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

#:alias An XML generator tag
Tag = tuple


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

#:alias ISO date/time like "2019-01-30 01:02:03"
DateTime = str

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
        class method:
            class Config:
                pass

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

    def __repr__(self):
        return repr(vars(self))

    def __getattr__(self, item):
        if item.startswith('_'):
            # do not use None fallback for special props
            raise AttributeError()
        return None

    def get(self, k, default=None):
        return vars(self).get(k, default)

    def _extend(self, args, kwargs):
        d = {}
        for a in args:
            if isinstance(a, dict):
                d.update(a)
            elif isinstance(a, Data):
                d.update(vars(a))
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


class Access(Config):
    """Access rights definition for authorization roles"""

    type: AccessType  #: access type (deny or allow)
    role: str  #: a role to which this rule applies


class WithAccess(Config):
    access: Optional[List[Access]]  #: access rights


class WithTypeAndAccess(Config):
    type: str  #: object type
    access: Optional[List[Access]]  #: access rights


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
    crs: 'Crs'
    extent: 'Extent'


class CorsOptions(Data):
    allow_credentials: bool
    allow_headers: Optional[List[str]]
    allow_origin: str


class DocumentRoot(Data):
    allow_mime: Optional[List[str]]
    deny_mime: Optional[List[str]]
    dir: 'DirPath'


class FeatureProps(Data):
    attributes: Optional[List[Attribute]]
    elements: Optional[dict]
    layerUid: Optional[str]
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']
    uid: Optional[str]


class IBaseRequest:
    data: Optional[bytes]
    environ: dict
    input_struct_type: int
    is_secure: bool
    method: str
    output_struct_type: int
    params: dict
    root: 'IRootObject'
    site: 'IWebSite'
    text: Optional[str]
    def cookie(self, key: str, default: str = None) -> str: pass
    def env(self, key: str, default: str = None) -> str: pass
    def error_response(self, err) -> 'IResponse': pass
    def file_response(self, path: str, mimetype: str, status: int = 200, attachment_name: str = None) -> 'IResponse': pass
    def has_param(self, key: str) -> bool: pass
    def header(self, key: str, default: str = None) -> str: pass
    def init(self): pass
    def param(self, key: str, default: str = None) -> str: pass
    def redirect_response(self, location, status=302): pass
    def response(self, content: str, mimetype: str, status: int = 200) -> 'IResponse': pass
    def struct_response(self, data: 'Response', status: int = 200) -> 'IResponse': pass
    def url_for(self, url: 'Url') -> 'Url': pass


class IFeature:
    attributes: List[Attribute]
    category: str
    data_model: Optional['IModel']
    elements: dict
    feature_format: Optional['IFormat']
    full_uid: str
    layer: Optional['ILayer']
    props: 'FeatureProps'
    props_for_render: 'FeatureProps'
    shape: Optional['IShape']
    style: Optional['IStyle']
    template_context: dict
    uid: str
    def apply_data_model(self, model: 'IModel' = None) -> 'IFeature': pass
    def apply_format(self, fmt: 'IFormat' = None, extra_context: dict = None, keys: List[str] = None) -> 'IFeature': pass
    def attr(self, name: str): pass
    def to_geojson(self) -> dict: pass
    def to_svg(self, rv: 'MapRenderView', style: 'IStyle' = None) -> str: pass
    def to_svg_tags(self, rv: 'MapRenderView', style: 'IStyle' = None) -> List['Tag']: pass
    def transform_to(self, crs) -> 'IFeature': pass


class IObject:
    access: 'Access'
    children: List['IObject']
    config: Config
    parent: 'IObject'
    props: Props
    root: 'IRootObject'
    uid: str
    def append_child(self, obj: 'IObject') -> 'IObject': pass
    def create_child(self, klass, cfg) -> 'IObject': pass
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
    def add_header(self, key, value): pass
    def delete_cookie(self, key, **kwargs): pass
    def set_cookie(self, key, **kwargs): pass


class IRole:
    def can_use(self, obj, parent=None): pass


class ISession:
    changed: bool
    data: dict
    method: 'IAuthMethod'
    type: str
    uid: str
    user: 'IUser'
    def get(self, key, default=None): pass
    def set(self, key, val): pass


class IShape:
    area: float
    bounds: 'Bounds'
    centroid: 'IShape'
    crs: str
    ewkb: bytes
    ewkb_hex: str
    ewkt: str
    extent: 'Extent'
    props: 'ShapeProps'
    srid: int
    type: 'GeometryType'
    wkb: bytes
    wkb_hex: str
    wkt: str
    x: float
    y: float
    def intersects(self, shape: 'IShape') -> bool: pass
    def to_multi(self) -> 'IShape': pass
    def to_type(self, new_type: 'GeometryType') -> 'IShape': pass
    def tolerance_polygon(self, tolerance, resolution=None) -> 'IShape': pass
    def transformed_to(self, to_crs, **kwargs) -> 'IShape': pass


class IStyle:
    name: str
    props: 'StyleProps'
    text: str
    type: 'StyleType'
    values: 'StyleValues'


class IUser:
    attributes: dict
    display_name: str
    fid: str
    is_guest: bool
    props: 'UserProps'
    provider: 'IAuthProvider'
    roles: List[str]
    uid: str
    def attribute(self, key: str, default: str = '') -> str: pass
    def can_use(self, obj, parent=None) -> bool: pass
    def has_role(self, role: str) -> bool: pass
    def init_from_data(self, provider, uid, roles, attributes) -> 'IUser': pass
    def init_from_source(self, provider, uid, roles=None, attributes=None) -> 'IUser': pass


class LayerLegend(Data):
    enabled: bool
    path: str
    template: 'ITemplate'
    url: str


class MapRenderInput(Data):
    background_color: int
    items: List['MapRenderInputItem']
    view: 'MapRenderView'


class MapRenderInputItem(Data):
    dpi: int
    features: List['IFeature']
    fragment: 'SvgFragment'
    layer: 'ILayer'
    opacity: float
    print_as_vector: bool
    style: 'IStyle'
    sub_layers: List[str]
    type: str


class MapRenderInputItemType(Enum):
    features = 'features'
    fragment = 'fragment'
    image = 'image'
    image_layer = 'image_layer'
    svg_layer = 'svg_layer'


class MapRenderOutput(Data):
    base_dir: str
    items: List['MapRenderOutputItem']
    view: 'MapRenderView'


class MapRenderOutputItem(Data):
    path: str
    tags: List['Tag']
    type: str


class MapRenderView(Data):
    bounds: 'Bounds'
    center: 'Point'
    dpi: int
    rotation: int
    scale: int
    size_mm: 'Size'
    size_px: 'Size'


class MetaContact(Data):
    address: str
    area: str
    city: str
    country: str
    email: str
    fax: str
    organization: str
    person: str
    phone: str
    position: str
    role: str
    url: str
    zip: str


class MetaData(Data):
    abstract: str
    accessConstraints: str
    attribution: str
    authorityIdentifier: str
    authorityName: str
    authorityUrl: 'Url'
    catalogUid: str
    contact: 'MetaContact'
    dateCreated: 'DateTime'
    dateUpdated: 'DateTime'
    fees: str
    image: 'Url'
    insipreKeywords: List['MetaInspireKeyword']
    insipreMandatoryKeyword: 'MetaInspireKeyword'
    inspireDegreeOfConformity: 'MetaInspireDegreeOfConformity'
    inspireResourceType: 'MetaInspireResourceType'
    inspireSpatialDataServiceType: 'MetaInspireSpatialDataServiceType'
    inspireTheme: 'MetaInspireTheme'
    isoMaintenanceFrequencyCode: 'MetaIsoMaintenanceFrequencyCode'
    isoQualityConformanceExplanation: str
    isoQualityConformancePass: bool
    isoQualityLineageSource: str
    isoQualityLineageSourceScale: int
    isoQualityLineageStatement: str
    isoScope: 'MetaIsoScope'
    isoSpatialRepresentationType: 'MetaIsoSpatialRepresentationType'
    isoTopicCategory: 'MetaIsoTopicCategory'
    keywords: List[str]
    language: str
    links: List['MetaLink']
    name: str
    serviceUrl: 'Url'
    title: str
    url: 'Url'
    urlType: str


class MetaInspireDegreeOfConformity(Enum):
    conformant = 'conformant'
    notConformant = 'notConformant'
    notEvaluated = 'notEvaluated'


class MetaInspireKeyword(Enum):
    chainDefinitionService = 'chainDefinitionService'
    comEncodingService = 'comEncodingService'
    comGeographicCompressionService = 'comGeographicCompressionService'
    comGeographicFormatConversionService = 'comGeographicFormatConversionService'
    comMessagingService = 'comMessagingService'
    comRemoteFileAndExecutableManagement = 'comRemoteFileAndExecutableManagement'
    comService = 'comService'
    comTransferService = 'comTransferService'
    humanCatalogueViewer = 'humanCatalogueViewer'
    humanChainDefinitionEditor = 'humanChainDefinitionEditor'
    humanFeatureGeneralizationEditor = 'humanFeatureGeneralizationEditor'
    humanGeographicDataStructureViewer = 'humanGeographicDataStructureViewer'
    humanGeographicFeatureEditor = 'humanGeographicFeatureEditor'
    humanGeographicSpreadsheetViewer = 'humanGeographicSpreadsheetViewer'
    humanGeographicSymbolEditor = 'humanGeographicSymbolEditor'
    humanGeographicViewer = 'humanGeographicViewer'
    humanInteractionService = 'humanInteractionService'
    humanServiceEditor = 'humanServiceEditor'
    humanWorkflowEnactmentManager = 'humanWorkflowEnactmentManager'
    infoCatalogueService = 'infoCatalogueService'
    infoCoverageAccessService = 'infoCoverageAccessService'
    infoFeatureAccessService = 'infoFeatureAccessService'
    infoFeatureTypeService = 'infoFeatureTypeService'
    infoGazetteerService = 'infoGazetteerService'
    infoManagementService = 'infoManagementService'
    infoMapAccessService = 'infoMapAccessService'
    infoOrderHandlingService = 'infoOrderHandlingService'
    infoProductAccessService = 'infoProductAccessService'
    infoRegistryService = 'infoRegistryService'
    infoSensorDescriptionService = 'infoSensorDescriptionService'
    infoStandingOrderService = 'infoStandingOrderService'
    metadataGeographicAnnotationService = 'metadataGeographicAnnotationService'
    metadataProcessingService = 'metadataProcessingService'
    metadataStatisticalCalculationService = 'metadataStatisticalCalculationService'
    spatialCoordinateConversionService = 'spatialCoordinateConversionService'
    spatialCoordinateTransformationService = 'spatialCoordinateTransformationService'
    spatialCoverageVectorConversionService = 'spatialCoverageVectorConversionService'
    spatialDimensionMeasurementService = 'spatialDimensionMeasurementService'
    spatialFeatureGeneralizationService = 'spatialFeatureGeneralizationService'
    spatialFeatureManipulationService = 'spatialFeatureManipulationService'
    spatialFeatureMatchingService = 'spatialFeatureMatchingService'
    spatialImageCoordinateConversionService = 'spatialImageCoordinateConversionService'
    spatialImageGeometryModelConversionService = 'spatialImageGeometryModelConversionService'
    spatialOrthorectificationService = 'spatialOrthorectificationService'
    spatialPositioningService = 'spatialPositioningService'
    spatialProcessingService = 'spatialProcessingService'
    spatialProximityAnalysisService = 'spatialProximityAnalysisService'
    spatialRectificationService = 'spatialRectificationService'
    spatialRouteDeterminationService = 'spatialRouteDeterminationService'
    spatialSamplingService = 'spatialSamplingService'
    spatialSensorGeometryModelAdjustmentService = 'spatialSensorGeometryModelAdjustmentService'
    spatialSubsettingService = 'spatialSubsettingService'
    spatialTilingChangeService = 'spatialTilingChangeService'
    subscriptionService = 'subscriptionService'
    taskManagementService = 'taskManagementService'
    temporalProcessingService = 'temporalProcessingService'
    temporalProximityAnalysisService = 'temporalProximityAnalysisService'
    temporalReferenceSystemTransformationService = 'temporalReferenceSystemTransformationService'
    temporalSamplingService = 'temporalSamplingService'
    temporalSubsettingService = 'temporalSubsettingService'
    thematicChangeDetectionService = 'thematicChangeDetectionService'
    thematicClassificationService = 'thematicClassificationService'
    thematicFeatureGeneralizationService = 'thematicFeatureGeneralizationService'
    thematicGeocodingService = 'thematicGeocodingService'
    thematicGeographicInformationExtractionService = 'thematicGeographicInformationExtractionService'
    thematicGeoparsingService = 'thematicGeoparsingService'
    thematicGoparameterCalculationService = 'thematicGoparameterCalculationService'
    thematicImageManipulationService = 'thematicImageManipulationService'
    thematicImageProcessingService = 'thematicImageProcessingService'
    thematicImageSynthesisService = 'thematicImageSynthesisService'
    thematicImageUnderstandingService = 'thematicImageUnderstandingService'
    thematicMultibandImageManipulationService = 'thematicMultibandImageManipulationService'
    thematicObjectDetectionService = 'thematicObjectDetectionService'
    thematicProcessingService = 'thematicProcessingService'
    thematicReducedResolutionGenerationService = 'thematicReducedResolutionGenerationService'
    thematicSpatialCountingService = 'thematicSpatialCountingService'
    thematicSubsettingService = 'thematicSubsettingService'
    workflowEnactmentService = 'workflowEnactmentService'


class MetaInspireResourceType(Enum):
    dataset = 'dataset'
    series = 'series'
    service = 'service'


class MetaInspireSpatialDataServiceType(Enum):
    discovery = 'discovery'
    download = 'download'
    invoke = 'invoke'
    other = 'other'
    transformation = 'transformation'
    view = 'view'


class MetaInspireTheme(Enum):
    ac = 'ac'
    ad = 'ad'
    af = 'af'
    am = 'am'
    au = 'au'
    br = 'br'
    bu = 'bu'
    cp = 'cp'
    ef = 'ef'
    el = 'el'
    er = 'er'
    ge = 'ge'
    gg = 'gg'
    gn = 'gn'
    hb = 'hb'
    hh = 'hh'
    hy = 'hy'
    lc = 'lc'
    lu = 'lu'
    mf = 'mf'
    mr = 'mr'
    nz = 'nz'
    of = 'of'
    oi = 'oi'
    pd = 'pd'
    pf = 'pf'
    ps = 'ps'
    rs = 'rs'
    sd = 'sd'
    so = 'so'
    sr = 'sr'
    su = 'su'
    tn = 'tn'
    us = 'us'


class MetaIsoMaintenanceFrequencyCode(Enum):
    annually = 'annually'
    asNeeded = 'asNeeded'
    biannually = 'biannually'
    continual = 'continual'
    daily = 'daily'
    fortnightly = 'fortnightly'
    irregular = 'irregular'
    monthly = 'monthly'
    notPlanned = 'notPlanned'
    quarterly = 'quarterly'
    unknown = 'unknown'
    weekly = 'weekly'


class MetaIsoOnLineFunction(Enum):
    download = 'download'
    information = 'information'
    offlineAccess = 'offlineAccess'
    order = 'order'
    search = 'search'


class MetaIsoScope(Enum):
    attribute = 'attribute'
    attributeType = 'attributeType'
    collectionHardware = 'collectionHardware'
    collectionSession = 'collectionSession'
    dataset = 'dataset'
    dimensionGroup = 'dimensionGroup'
    feature = 'feature'
    featureType = 'featureType'
    fieldSession = 'fieldSession'
    initiative = 'initiative'
    model = 'model'
    nonGeographicDataset = 'nonGeographicDataset'
    otherAggregate = 'otherAggregate'
    platformSeries = 'platformSeries'
    productionSeries = 'productionSeries'
    propertyType = 'propertyType'
    sensor = 'sensor'
    sensorSeries = 'sensorSeries'
    series = 'series'
    service = 'service'
    software = 'software'
    stereomate = 'stereomate'
    tile = 'tile'
    transferAggregate = 'transferAggregate'


class MetaIsoSpatialRepresentationType(Enum):
    grid = 'grid'
    stereoModel = 'stereoModel'
    textTable = 'textTable'
    tin = 'tin'
    vector = 'vector'
    video = 'video'


class MetaIsoTopicCategory(Enum):
    biota = 'biota'
    boundaries = 'boundaries'
    climatologyMeteorologyAtmosphere = 'climatologyMeteorologyAtmosphere'
    economy = 'economy'
    elevation = 'elevation'
    environment = 'environment'
    farming = 'farming'
    geoscientificInformation = 'geoscientificInformation'
    health = 'health'
    imageryBaseMapsEarthCover = 'imageryBaseMapsEarthCover'
    inlandWaters = 'inlandWaters'
    intelligenceMilitary = 'intelligenceMilitary'
    location = 'location'
    oceans = 'oceans'
    planningCadastre = 'planningCadastre'
    society = 'society'
    structure = 'structure'
    transportation = 'transportation'
    utilitiesCommunication = 'utilitiesCommunication'


class MetaLink(Data):
    function: 'MetaIsoOnLineFunction'
    scheme: str
    url: 'Url'


class ModelProps(Props):
    rules: List['ModelRule']


class ModelRule(Data):
    editable: bool
    expression: str
    format: 'FormatStr'
    name: str
    source: str
    title: str
    type: 'AttributeType'
    value: Optional[str]


class OwsOperation:
    formats: List[str]
    get_url: 'Url'
    name: str
    parameters: dict
    post_url: 'Url'


class Projection(Data):
    epsg: str
    is_geographic: bool
    proj4text: str
    srid: int
    units: str
    uri: str
    url: str
    urn: str
    urnx: str


class RewriteRule(Data):
    match: 'Regex'
    options: Optional[dict]
    target: str


class SearchArgs(Data):
    axis: str
    bounds: 'Bounds'
    keyword: Optional[str]
    layers: List['ILayer']
    limit: int
    params: dict
    project: 'IProject'
    resolution: float
    shapes: List['IShape']
    source_layer_names: List[str]
    tolerance: 'Measurement'


class SelectArgs(Data):
    extra_where: Optional[list]
    keyword: Optional[str]
    limit: Optional[int]
    map_tolerance: Optional[float]
    shape: Optional['IShape']
    sort: Optional[str]
    table: 'SqlTable'
    uids: Optional[List[str]]


class ShapeProps(Props):
    crs: str
    geometry: dict


class SourceLayer(Data):
    a_level: int
    a_path: str
    a_uid: str
    data_source: dict
    is_expanded: bool
    is_group: bool
    is_image: bool
    is_queryable: bool
    is_visible: bool
    layers: List['SourceLayer']
    legend: str
    meta: 'MetaData'
    name: str
    opacity: int
    resource_urls: dict
    scale_range: List[float]
    styles: List['SourceStyle']
    supported_bounds: List['Bounds']
    supported_crs: List['Crs']
    title: str


class SourceStyle(Data):
    is_default: bool
    legend: 'Url'
    meta: 'MetaData'


class SpecValidator:
    def method_spec(self, name): pass
    def read_value(self, val, type_name, path='', strict=True): pass


class SqlTable(Data):
    geometry_column: str
    geometry_crs: 'Crs'
    geometry_type: 'GeometryType'
    key_column: str
    name: str
    search_column: str


class SqlTableColumn(Data):
    crs: 'Crs'
    geom_type: 'GeometryType'
    is_geometry: bool
    is_key: bool
    name: str
    native_type: str
    type: 'AttributeType'


class StorageDirectory(Data):
    category: str
    entries: List['StorageEntry']
    readable: bool
    writable: bool


class StorageElement(Data):
    data: dict
    entry: 'StorageEntry'


class StorageEntry(Data):
    category: str
    name: str


class StorageRecord(Data):
    category: str
    created: int
    data: str
    name: str
    updated: int
    user_fid: str


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
    name: Optional[str]
    text: Optional[str]
    type: 'StyleType'
    values: Optional['StyleValues']


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
    fill: Optional['Color']
    icon: Optional[str]
    label_align: Optional['StyleLabelAlign']
    label_background: Optional['Color']
    label_fill: Optional['Color']
    label_font_family: Optional[str]
    label_font_size: Optional[int]
    label_font_style: Optional['StyleLabelFontStyle']
    label_font_weight: Optional['StyleLabelFontWeight']
    label_line_height: Optional[int]
    label_max_scale: Optional[int]
    label_min_scale: Optional[int]
    label_offset_x: Optional[int]
    label_offset_y: Optional[int]
    label_padding: Optional[List[int]]
    label_placement: Optional['StyleLabelPlacement']
    label_stroke: Optional['Color']
    label_stroke_dasharray: Optional[List[int]]
    label_stroke_dashoffset: Optional[int]
    label_stroke_linecap: Optional['StyleStrokeLineCap']
    label_stroke_linejoin: Optional['StyleStrokeLineJoin']
    label_stroke_miterlimit: Optional[int]
    label_stroke_width: Optional[int]
    marker: Optional['StyleMarker']
    marker_fill: Optional['Color']
    marker_size: Optional[int]
    marker_stroke: Optional['Color']
    marker_stroke_dasharray: Optional[List[int]]
    marker_stroke_dashoffset: Optional[int]
    marker_stroke_linecap: Optional['StyleStrokeLineCap']
    marker_stroke_linejoin: Optional['StyleStrokeLineJoin']
    marker_stroke_miterlimit: Optional[int]
    marker_stroke_width: Optional[int]
    offset_x: Optional[int]
    offset_y: Optional[int]
    point_size: Optional[int]
    stroke: Optional['Color']
    stroke_dasharray: Optional[List[int]]
    stroke_dashoffset: Optional[int]
    stroke_linecap: Optional['StyleStrokeLineCap']
    stroke_linejoin: Optional['StyleStrokeLineJoin']
    stroke_miterlimit: Optional[int]
    stroke_width: Optional[int]
    with_geometry: Optional['StyleGeometryOption']
    with_label: Optional['StyleLabelOption']


class SvgFragment(Data):
    points: List['Point']
    styles: Optional[List['IStyle']]
    tags: List['Tag']


class TemplateLegendMode(Enum):
    html = 'html'
    image = 'image'


class TemplateOutput(Data):
    content: str
    mime: str
    path: str


class TemplateProps(Props):
    dataModel: 'ModelProps'
    mapHeight: int
    mapWidth: int
    qualityLevels: List['TemplateQualityLevel']
    title: str
    uid: str


class TemplateQualityLevel(Data):
    dpi: int
    name: str


class UserProps(Data):
    displayName: str


class IApi(IObject):
    actions: dict


class IApplication(IObject):
    api: 'IApi'
    auth: 'IAuthManager'
    client: Optional['IClient']
    meta: 'MetaData'
    monitor: 'IMonitor'
    qgis_version: str
    version: str
    web_sites: List['IWebSite']
    def developer_option(self, name): pass
    def find_action(self, action_type, project_uid=None): pass


class IAuthManager(IObject):
    guest_user: 'IUser'
    methods: List['IAuthMethod']
    providers: List['IAuthProvider']
    def authenticate(self, method: 'IAuthMethod', login, password, **kw) -> Optional['IUser']: pass
    def close_session(self, sess: 'ISession', req: 'IRequest', res: 'IResponse') -> 'ISession': pass
    def create_stored_session(self, type: str, method: 'IAuthMethod', user: 'IUser') -> 'ISession': pass
    def delete_stored_sessions(self): pass
    def destroy_stored_session(self, sess: 'ISession'): pass
    def find_stored_session(self, uid): pass
    def get_method(self, type: str) -> Optional['IAuthMethod']: pass
    def get_provider(self, uid: str) -> Optional['IAuthProvider']: pass
    def get_role(self, name: str) -> 'IRole': pass
    def get_user(self, user_fid: str) -> Optional['IUser']: pass
    def login(self, method: 'IAuthMethod', login: str, password: str, req: 'IRequest') -> 'ISession': pass
    def logout(self, sess: 'ISession', req: 'IRequest') -> 'ISession': pass
    def new_session(self, **kwargs): pass
    def open_session(self, req: 'IRequest') -> 'ISession': pass
    def save_stored_session(self, sess: 'ISession'): pass
    def serialize_user(self, user: 'IUser') -> str: pass
    def stored_session_records(self) -> List[dict]: pass
    def unserialize_user(self, s: str) -> 'IUser': pass


class IAuthMethod(IObject):
    type: str
    def close_session(self, auth: 'IAuthManager', sess: 'ISession', req: 'IRequest', res: 'IResponse'): pass
    def login(self, auth: 'IAuthManager', login: str, password: str, req: 'IRequest') -> Optional['ISession']: pass
    def logout(self, auth: 'IAuthManager', sess: 'ISession', req: 'IRequest') -> 'ISession': pass
    def open_session(self, auth: 'IAuthManager', req: 'IRequest') -> Optional['ISession']: pass


class IAuthProvider(IObject):
    allowed_methods: List[str]
    def authenticate(self, method: 'IAuthMethod', login: str, password: str, **kwargs) -> Optional['IUser']: pass
    def get_user(self, user_uid: str) -> Optional['IUser']: pass
    def user_from_dict(self, d: dict) -> 'IUser': pass
    def user_to_dict(self, u: 'IUser') -> dict: pass


class IClient(IObject):
    pass


class IDbProvider(IObject):
    pass


class IFormat(IObject):
    templates: dict
    def apply(self, context: dict, keys: List[str] = None) -> dict: pass


class ILayer(IObject):
    cache_uid: str
    can_render_box: bool
    can_render_svg: bool
    can_render_xyz: bool
    crs: str
    data_model: Optional['IModel']
    default_search_provider: Optional['ISearchProvider']
    description: str
    description_template: 'ITemplate'
    display: str
    edit_data_model: Optional['IModel']
    edit_options: Data
    edit_style: Optional['IStyle']
    extent: Optional['Extent']
    feature_format: 'IFormat'
    geometry_type: Optional['GeometryType']
    grid_uid: str
    has_cache: bool
    has_legend: bool
    has_search: bool
    image_format: str
    is_editable: bool
    is_group: bool
    is_public: bool
    layers: List['ILayer']
    legend: 'LayerLegend'
    map: 'IMap'
    meta: 'MetaData'
    opacity: float
    own_bounds: Optional['Bounds']
    ows_name: str
    resolutions: List[float]
    style: 'IStyle'
    supports_wfs: bool
    supports_wms: bool
    title: str
    def configure_legend(self) -> 'LayerLegend': pass
    def configure_metadata(self, provider_meta=None) -> 'MetaData': pass
    def configure_search(self): pass
    def edit_access(self, user): pass
    def edit_operation(self, operation: str, feature_props: List['FeatureProps']) -> List['IFeature']: pass
    def get_features(self, bounds: 'Bounds', limit: int = 0) -> List['IFeature']: pass
    def mapproxy_config(self, mc): pass
    def ows_enabled(self, service: 'IOwsService') -> bool: pass
    def render_box(self, rv: 'MapRenderView', extra_params=None): pass
    def render_html_legend(self, context=None) -> str: pass
    def render_legend(self, context=None) -> Optional[str]: pass
    def render_legend_image(self, context=None) -> bytes: pass
    def render_svg(self, rv: 'MapRenderView', style: 'IStyle' = None) -> str: pass
    def render_svg_tags(self, rv: 'MapRenderView', style: 'IStyle' = None) -> List['Tag']: pass
    def render_xyz(self, x, y, z): pass


class IMap(IObject):
    bounds: 'Bounds'
    center: 'Point'
    coordinate_precision: float
    crs: 'Crs'
    extent: 'Extent'
    init_resolution: float
    layers: List['ILayer']
    resolutions: List[float]


class IModel(IObject):
    attribute_names: List[str]
    geometry_crs: 'Crs'
    geometry_type: 'GeometryType'
    rules: List['ModelRule']
    def apply(self, atts: List[Attribute]) -> List[Attribute]: pass
    def apply_to_dict(self, d: dict) -> List[Attribute]: pass


class IMonitor(IObject):
    path_stats: dict
    watch_dirs: dict
    watch_files: dict
    def add_directory(self, path, pattern): pass
    def add_path(self, path): pass
    def start(self): pass


class IOwsProvider(IObject):
    invert_axis_crs: List[str]
    meta: 'MetaData'
    operations: List['OwsOperation']
    source_layers: List['SourceLayer']
    supported_crs: List['Crs']
    type: str
    url: 'Url'
    version: str
    def find_features(self, args: 'SearchArgs') -> List['IFeature']: pass
    def operation(self, name: str) -> 'OwsOperation': pass


class IOwsService(IObject):
    meta: 'MetaData'
    type: str
    version: str
    def error_response(self, err: 'Exception') -> 'HttpResponse': pass
    def handle(self, req: 'IRequest') -> 'HttpResponse': pass


class IPrinter(IObject):
    templates: List['ITemplate']


class IProject(IObject):
    api: Optional['IApi']
    assets_root: Optional['DocumentRoot']
    client: Optional['IClient']
    description_template: 'ITemplate'
    locales: List[str]
    map: Optional['IMap']
    meta: 'MetaData'
    overview_map: Optional['IMap']
    printer: Optional['IPrinter']
    title: str


class IRequest(IBaseRequest):
    auth: 'IAuthManager'
    session: 'ISession'
    user: 'IUser'
    def acquire(self, klass: str, uid: str) -> Optional['IObject']: pass
    def auth_close(self, res: 'IResponse'): pass
    def auth_open(self): pass
    def login(self, login: str, password: str): pass
    def logout(self): pass
    def require(self, klass: str, uid: str) -> 'IObject': pass
    def require_layer(self, uid: str) -> 'ILayer': pass
    def require_project(self, uid: str) -> 'IProject': pass


class IRootObject(IObject):
    all_objects: list
    all_types: dict
    application: 'IApplication'
    shared_objects: dict
    validator: 'SpecValidator'
    def create(self, klass, cfg=None): pass
    def create_object(self, klass, cfg, parent=None): pass
    def create_shared_object(self, klass, uid, cfg): pass
    def create_unbound_object(self, klass, cfg): pass
    def find(self, klass, uid=None) -> 'IObject': pass
    def find_all(self, klass=None) -> List['IObject']: pass
    def find_by_uid(self, uid) -> 'IObject': pass
    def find_first(self, klass) -> 'IObject': pass


class ISearchProvider(IObject):
    active: bool
    data_model: Optional['IModel']
    feature_format: Optional['IFormat']
    tolerance: 'Measurement'
    with_geometry: bool
    with_keyword: bool
    def can_run(self, args: 'SearchArgs'): pass
    def context_shape(self, args: 'SearchArgs') -> 'IShape': pass
    def run(self, layer: 'ILayer', args: 'SearchArgs') -> List['IFeature']: pass


class ITemplate(IObject):
    data_model: Optional['IModel']
    legend_layer_uids: List[str]
    legend_mode: Optional['TemplateLegendMode']
    map_size: 'Size'
    page_size: 'Size'
    path: str
    text: str
    title: str
    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str: pass
    def dpi_for_quality(self, quality): pass
    def normalize_context(self, context: dict) -> dict: pass
    def render(self, context: dict, mro: 'MapRenderOutput' = None, out_path: str = None, legends: dict = None, format: str = None) -> 'TemplateOutput': pass


class IWebSite(IObject):
    assets_root: 'DocumentRoot'
    cors: 'CorsOptions'
    error_page: Optional['ITemplate']
    host: str
    reversed_host: str
    reversed_rewrite_rules: List['RewriteRule']
    rewrite_rules: List['RewriteRule']
    ssl: bool
    static_root: 'DocumentRoot'
    def url_for(self, req, url): pass


class ISqlProvider(IDbProvider):
    def describe(self, table: 'SqlTable') -> Dict[str, 'SqlTableColumn']: pass
    def edit_operation(self, operation: str, table: 'SqlTable', features: List['IFeature']) -> List['IFeature']: pass
    def select(self, args: 'SelectArgs', extra_connect_params: dict = None) -> List['IFeature']: pass


class IVectorLayer(ILayer):
    def connect_feature(self, feature: 'IFeature') -> 'IFeature': pass
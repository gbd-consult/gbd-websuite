"""Utilities to manipulate metadata"""

import gws
import gws.tools.country
import gws.tools.date

import gws.types as t

from . import inspire


#:export
class MetaInspireTheme(t.Enum):
    """Inspire theme, see http://inspire.ec.europa.eu/theme/"""
    ac = 'ac'  #: Atmospheric conditions
    ad = 'ad'  #: Addresses
    af = 'af'  #: Agricultural and aquaculture facilities
    am = 'am'  #: Area management/restriction/regulation zones and reporting units
    au = 'au'  #: Administrative units
    br = 'br'  #: Bio-geographical regions
    bu = 'bu'  #: Buildings
    cp = 'cp'  #: Cadastral parcels
    ef = 'ef'  #: Environmental monitoring facilities
    el = 'el'  #: Elevation
    er = 'er'  #: Energy resources
    ge = 'ge'  #: Geology
    gg = 'gg'  #: Geographical grid systems
    gn = 'gn'  #: Geographical names
    hb = 'hb'  #: Habitats and biotopes
    hh = 'hh'  #: Human health and safety
    hy = 'hy'  #: Hydrography
    lc = 'lc'  #: Land cover
    lu = 'lu'  #: Land use
    mf = 'mf'  #: Meteorological geographical features
    mr = 'mr'  #: Mineral resources
    nz = 'nz'  #: Natural risk zones
    of = 'of'  #: Oceanographic geographical features
    oi = 'oi'  #: Orthoimagery
    pd = 'pd'  #: Population distribution — demography
    pf = 'pf'  #: Production and industrial facilities
    ps = 'ps'  #: Protected sites
    rs = 'rs'  #: Coordinate reference systems
    sd = 'sd'  #: Species distribution
    so = 'so'  #: Soil
    sr = 'sr'  #: Sea regions
    su = 'su'  #: Statistical units
    tn = 'tn'  #: Transport networks
    us = 'us'  #: Utility and governmental services


#:export
class MetaInspireResourceType(t.Enum):
    """Inspire resourceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    dataset = 'dataset'
    series = 'series'
    service = 'service'


#:export
class MetaInspireSpatialDataServiceType(t.Enum):
    """Inspire spatialDataServiceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    discovery = 'discovery'
    view = 'view'
    download = 'download'
    transformation = 'transformation'
    invoke = 'invoke'
    other = 'other'


#:export
class MetaInspireDegreeOfConformity(t.Enum):
    """Inspire degreeOfConformity, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    conformant = 'conformant'
    notConformant = 'notConformant'
    notEvaluated = 'notEvaluated'


#:export
class MetaInspireMandatoryKeyword(t.Enum):
    """Inspire keyword, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
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


#:export
class MetaIsoScope(t.Enum):
    """ISO-19139 MD/MX_ScopeCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    attribute = 'attribute'  #: information applies to the attribute class
    attributeType = 'attributeType'  #: information applies to the characteristic of a feature
    collectionHardware = 'collectionHardware'  #: information applies to the collection hardware class
    collectionSession = 'collectionSession'  #: information applies to the collection session
    dataset = 'dataset'  #: information applies to the dataset
    series = 'series'  #: information applies to the series
    nonGeographicDataset = 'nonGeographicDataset'  #: information applies to non-geographic data
    dimensionGroup = 'dimensionGroup'  #: information applies to a dimension group
    feature = 'feature'  #: information applies to a feature
    featureType = 'featureType'  #: information applies to a feature type
    propertyType = 'propertyType'  #: information applies to a property type
    fieldSession = 'fieldSession'  #: information applies to a field session
    software = 'software'  #: information applies to a computer program or routine
    service = 'service'  #: information applies to a capability which a service provider entity makes available to a service user entity through a set of interfaces that define a behaviour, such as a use case
    model = 'model'  #: information applies to a copy or imitation of an existing or hypothetical object
    tile = 'tile'  #: information applies to a tile, a spatial subset of geographic data
    initiative = 'initiative'  #: The referencing entity applies to a transfer aggregate which was originally identified as an initiative (DS_Initiative)
    stereomate = 'stereomate'  #: The referencing entity applies to a transfer aggregate which was originally identified as a stereo mate (DS_StereoMate)
    sensor = 'sensor'  #: The referencing entity applies to a transfer aggregate which was originally identified as a sensor (DS_Sensor)
    platformSeries = 'platformSeries'  #: The referencing entity applies to a transfer aggregate which was originally identified as a platform series (DS_PlatformSeries)
    sensorSeries = 'sensorSeries'  #: The referencing entity applies to a transfer aggregate which was originally identified as a sensor series (DS_SensorSeries)
    productionSeries = 'productionSeries'  #: The referencing entity applies to a transfer aggregate which was originally identified as a production series (DS_ProductionSeries)
    transferAggregate = 'transferAggregate'  #: The referencing entity applies to a transfer aggregate which has no existence outside of the transfer context
    otherAggregate = 'otherAggregate'  #: The referencing entity applies to a transfer aggregate which has an existence outside of the transfer context, but which does not pertains to a specific aggregate type


#:export
class MetaIsoMaintenanceFrequencyCode(t.Enum):
    """ISO-19139 MD_MaintenanceFrequencyCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    continual = 'continual'
    daily = 'daily'
    weekly = 'weekly'
    fortnightly = 'fortnightly'
    monthly = 'monthly'
    quarterly = 'quarterly'
    biannually = 'biannually'
    annually = 'annually'
    asNeeded = 'asNeeded'
    irregular = 'irregular'
    notPlanned = 'notPlanned'
    unknown = 'unknown'


#:export
class MetaIsoSpatialRepresentationType(t.Enum):
    """ISO-19139 MD_SpatialRepresentationTypeCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    vector = 'vector'  #: vector data is used to represent geographic data
    grid = 'grid'  #: grid data is used to represent geographic data
    textTable = 'textTable'  #: textual or tabular data is used to represent geographic data
    tin = 'tin'  #: triangulated irregular network
    stereoModel = 'stereoModel'  #: three-dimensional view formed by the intersecting homologous rays of an overlapping pair of images
    video = 'video'  #: scene from a video recording


#:export
class MetaIsoOnLineFunction(t.Enum):
    """ISO-19139 CI_OnLineFunctionCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    download = 'download'  #: online instructions for transferring data from one storage device or system to another
    information = 'information'  #: online information about the resource
    offlineAccess = 'offlineAccess'  #: online instructions for requesting the resource from the provider
    order = 'order'  #: online order process for obtening the resource
    search = 'search'  #: online search interface for seeking out information about the resource


#:export
class MetaIsoTopicCategory(t.Enum):
    """ISO-19139 MD_TopicCategoryCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    farming = 'farming'  #: rearing of animals and/or cultivation of plants. Examples: agriculture, irrigation, aquaculture, plantations, herding, pests and diseases affecting crops and livestock
    biota = 'biota'  #: flora and/or fauna in natural environment. Examples: wildlife, vegetation, biological sciences, ecology, wilderness, sealife, wetlands, habitat
    boundaries = 'boundaries'  #: legal land descriptions. Examples: political and administrative boundaries
    climatologyMeteorologyAtmosphere = 'climatologyMeteorologyAtmosphere'  #: processes and phenomena of the atmosphere. Examples: cloud cover, weather, climate, atmospheric conditions, climate change, precipitation
    economy = 'economy'  #: economic activities, conditions and employment. Examples: production, labour, revenue, commerce, industry, tourism and ecotourism, forestry, fisheries, commercial or subsistence hunting, exploration and exploitation of resources such as minerals, oil and gas
    elevation = 'elevation'  #: height above or below sea level. Examples: altitude, bathymetry, digital elevation models, slope, derived products
    environment = 'environment'  #: environmental resources, protection and conservation. Examples: environmental pollution, waste storage and treatment, environmental impact assessment, monitoring environmental risk, nature reserves, landscape
    geoscientificInformation = 'geoscientificInformation'  #: information pertaining to earth sciences. Examples: geophysical features and processes, geology, minerals, sciences dealing with the composition, structure and origin of the earth s rocks, risks of earthquakes, volcanic activity, landslides, gravity information, soils, permafrost, hydrogeology, erosion
    health = 'health'  #: health, health services, human ecology, and safety. Examples: disease and illness, factors affecting health, hygiene, substance abuse, mental and physical health, health services
    imageryBaseMapsEarthCover = 'imageryBaseMapsEarthCover'  #: base maps. Examples: land cover, topographic maps, imagery, unclassified images, annotations
    intelligenceMilitary = 'intelligenceMilitary'  #: military bases, structures, activities. Examples: barracks, training grounds, military transportation, information collection
    inlandWaters = 'inlandWaters'  #: inland water features, drainage systems and their characteristics. Examples: rivers and glaciers, salt lakes, water utilization plans, dams, currents, floods, water quality, hydrographic charts
    location = 'location'  #: positional information and services. Examples: addresses, geodetic networks, control points, postal zones and services, place names
    oceans = 'oceans'  #: features and characteristics of salt water bodies (excluding inland waters). Examples: tides, tidal waves, coastal information, reefs
    planningCadastre = 'planningCadastre'  #: information used for appropriate actions for future use of the land. Examples: land use maps, zoning maps, cadastral surveys, land ownership
    society = 'society'  #: characteristics of society and cultures. Examples: settlements, anthropology, archaeology, education, traditional beliefs, manners and customs, demographic data, recreational areas and activities, social impact assessments, crime and justice, census information
    structure = 'structure'  #: man-made construction. Examples: buildings, museums, churches, factories, housing, monuments, shops, towers
    transportation = 'transportation'  #: means and aids for conveying persons and/or goods. Examples: roads, airports/airstrips, shipping routes, tunnels, nautical charts, vehicle or vessel location, aeronautical charts, railways
    utilitiesCommunication = 'utilitiesCommunication'  #: energy, water and waste systems and communications infrastructure and services. Examples: hydroelectricity, geothermal, solar and nuclear sources of energy, water purification and distribution, sewage collection and disposal, electricity and gas distribution, data communication, telecommunication, radio, communication networks


#:export
class MetaIsoRestrictionCode(t.Enum):
    """ISO-19139 MD_RestrictionCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml"""
    copyright = 'copyright'  #: exclusive right to the publication, production, or sale of the rights to a literary, dramatic, musical, or artistic work, or to the use of a commercial print or label, granted by law for a specified period of time to an author, composer, artist, distributor
    patent = 'patent'  #: government has granted exclusive right to make, sell, use or license an invention or discovery
    patentPending = 'patentPending'  #: produced or sold information awaiting a patent
    trademark = 'trademark'  #: a name, symbol, or other device identifying a product, officially registered and legally restricted to the use of the owner or manufacturer
    license = 'license'  #: formal permission to do something
    intellectualPropertyRights = 'intellectualPropertyRights'  #: rights to financial benefit from and control of distribution of non-tangible property that is a result of creativity
    restricted = 'restricted'  #: withheld from general circulation or disclosure
    otherRestrictions = 'otherRestrictions'  #: limitation not listed


#:export
class MetaIsoQualityConformance(t.Data):
    specificationTitle: str
    specificationDate: str
    explanation: str = ''
    qualityPass: bool


#:export
class MetaIsoQualityLineage(t.Data):
    statement: str
    source: str
    sourceScale: int


class ContactConfig(t.Config):
    """Contact metadata configuration"""

    address: t.Optional[str]
    area: t.Optional[str]
    city: t.Optional[str]
    country: t.Optional[str]
    email: t.Optional[str]
    fax: t.Optional[str]
    organization: t.Optional[str]
    person: t.Optional[str]
    phone: t.Optional[str]
    position: t.Optional[str]
    zip: t.Optional[str]
    url: t.Url = ''


class LinkConfig(t.Config):
    """Object link configuration"""

    scheme: t.Optional[str]  #: link scheme
    url: t.Url  #: link url
    function: t.Optional[MetaIsoOnLineFunction]  #: ISO-19115 function


class Config(t.Config):
    """Object metadata configuration"""

    abstract: t.Optional[str]  #: object abstract description
    accessConstraints: t.Optional[str]
    license: t.Optional[str]
    attribution: t.Optional[str]  #: attribution (copyright) string
    contact: t.Optional[ContactConfig]  #: contact information
    dateCreated: t.Optional[t.Date]  #: publication date
    dateUpdated: t.Optional[t.Date]  #: modification date
    dateBegin: t.Optional[t.Date]  #: temporal extent begin
    dateEnd: t.Optional[t.Date]  #: temporal extent end
    fees: t.Optional[str]
    image: t.Optional[t.Url]  #: image (logo) url

    authorityName: t.Optional[str]
    authorityUrl: t.Optional[t.Url]
    authorityIdentifier: t.Optional[str]

    insipreKeywords: t.Optional[t.List[MetaInspireMandatoryKeyword]]
    insipreMandatoryKeyword: t.Optional[MetaInspireMandatoryKeyword]
    inspireDegreeOfConformity: t.Optional[MetaInspireDegreeOfConformity]
    inspireResourceType: t.Optional[MetaInspireResourceType]
    inspireSpatialDataServiceType: t.Optional[MetaInspireSpatialDataServiceType]
    inspireTheme: t.Optional[MetaInspireTheme]

    isoMaintenanceFrequencyCode: t.Optional[MetaIsoMaintenanceFrequencyCode]
    isoScope: t.Optional[MetaIsoScope]  #: ISO-19139 scope code
    isoScopeName: t.Optional[str]  #: ISO-19139 scope name
    isoSpatialRepresentationType: t.Optional[MetaIsoSpatialRepresentationType]  #: ISO-19139 spatial type
    isoTopicCategory: t.Optional[MetaIsoTopicCategory]  #: ISO-19139 topic category

    isoQualityConformance: t.Optional[t.List[MetaIsoQualityConformance]]
    isoQualityLineage: t.Optional[MetaIsoQualityLineage]

    isoRestrictionCode: t.Optional[MetaIsoRestrictionCode]

    catalogUid: t.Optional[str]  #: catalog identifier

    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    links: t.List[LinkConfig] = []  #: additional links
    name: t.Optional[str]  #: object internal name
    serviceUrl: t.Optional[t.Url]  #: service url
    title: t.Optional[str]  #: object title
    url: t.Optional[t.Url]  #: metadata url
    urlType: t.Optional[str]  #: metadata url type like "TC211"
    urlFormat: t.Optional[str]  #: metadata url mime type


#:export
class MetaContact(t.Data):
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
    zip: str
    url: str


#:export
class MetaLink(t.Data):
    scheme: str
    url: t.Url
    function: MetaIsoOnLineFunction


#:export
class MetaData(t.Data):
    abstract: str
    accessConstraints: str
    license: str
    attribution: str
    contact: MetaContact
    dateCreated: t.DateTime
    dateUpdated: t.DateTime
    dateBegin: t.DateTime
    dateEnd: t.DateTime
    fees: str
    image: t.Url

    authorityName: str
    authorityUrl: t.Url
    authorityIdentifier: str

    insipreKeywords: t.List[MetaInspireMandatoryKeyword]
    insipreMandatoryKeyword: MetaInspireMandatoryKeyword
    inspireDegreeOfConformity: MetaInspireDegreeOfConformity
    inspireResourceType: MetaInspireResourceType
    inspireSpatialDataServiceType: MetaInspireSpatialDataServiceType
    inspireTheme: MetaInspireTheme
    inspireThemeName: str
    inspireThemeNameEn: str

    isoMaintenanceFrequencyCode: MetaIsoMaintenanceFrequencyCode
    isoScope: MetaIsoScope
    isoScopeName: str
    isoSpatialRepresentationType: MetaIsoSpatialRepresentationType
    isoTopicCategory: MetaIsoTopicCategory

    isoQualityConformance: MetaIsoQualityConformance
    isoQualityLineage: MetaIsoQualityLineage
    isoRestrictionCode: str

    catalogUid: str

    keywords: t.List[str]
    language: str
    links: t.List[MetaLink]
    name: str
    serviceUrl: t.Url
    title: str

    url: t.Url
    urlType: str
    urlFormat: str


class Props(t.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: t.List[str]
    language: str
    title: str


def props(m: t.MetaData) -> Props:
    return Props(
        abstract=m.abstract or '',
        attribution=m.attribution or '',
        dateCreated=m.dateCreated,
        dateUpdated=m.dateUpdated,
        keywords=m.keywords or '',
        language=m.language or '',
        title=m.title or '',
    )


def from_config(m: t.Config) -> t.MetaData:
    if not m:
        return t.MetaData()

    meta = t.MetaData(m)

    if meta.language:
        meta.language3 = gws.tools.country.bibliographic_name(language=meta.language)

    meta.contact = MetaContact(meta.contact or {})

    if meta.keywords:
        meta.keywords = gws.strip(meta.keywords) or None

    if meta.inspireTheme:
        meta.inspireThemeName = inspire.theme_name(meta.inspireTheme, meta.language)
        meta.inspireThemeNameEn = inspire.theme_name(meta.inspireTheme, 'en')

    return meta


def from_dict(d: dict) -> t.MetaData:
    m = {}
    contact = None

    for k, v in d.items():
        if k.startswith('contact.'):
            contact = contact or {}
            contact[k.split('.')[1]] = v
        else:
            m[k] = v

    cfg = t.Config(m)
    if contact:
        cfg.contact = t.Config(contact)

    return from_config(cfg)


def from_meta(m: t.MetaData) -> t.MetaData:
    return from_config(t.cast(t.Config, m))


def extend(a: t.MetaData, b: t.MetaData):
    a = gws.extend(a, b)
    if gws.has(b, 'contact'):
        a.contact = gws.extend(a.contact, b.contact)
    kwa = gws.get(a, 'keywords') or []
    kwb = gws.get(b, 'keywords') or []
    a.keywords = sorted(set(kwa + kwb))
    return a

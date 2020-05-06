"""Utilities to manipulate metadata"""

import gws
import gws.tools.country

import gws.types as t

from . import inspire


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
class MetaInspireKeyword(t.Enum):
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
    """ISO-19139 MD/MX_ScopeCode, see http://schemas.opengis.net/iso/19139/20070417/resources/codelist/gmxCodelists.xml"""
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
class MetaIsoSpatialRepresentationType(t.Enum):
    """ISO-19139 MD_SpatialRepresentationTypeCode, see http://schemas.opengis.net/iso/19139/20070417/resources/codelist/gmxCodelists.xml"""
    vector = 'vector'  #: vector data is used to represent geographic data
    grid = 'grid'  #: grid data is used to represent geographic data
    textTable = 'textTable'  #: textual or tabular data is used to represent geographic data
    tin = 'tin'  #: triangulated irregular network
    stereoModel = 'stereoModel'  #: three-dimensional view formed by the intersecting homologous rays of an overlapping pair of images
    video = 'video'  #: scene from a video recording


#:export
class MetaIsoOnLineFunction(t.Enum):
    """ISO-19139 CI_OnLineFunctionCode, see http://schemas.opengis.net/iso/19139/20070417/resources/codelist/gmxCodelists.xml"""
    download = 'download'  #: online instructions for transferring data from one storage device or system to another
    information = 'information'  #: online information about the resource
    offlineAccess = 'offlineAccess'  #: online instructions for requesting the resource from the provider
    order = 'order'  #: online order process for obtening the resource
    search = 'search'  #: online search interface for seeking out information about the resource


#:export
class MetaIsoTopicCategory(t.Enum):
    """ISO-19139 MD_TopicCategoryCode, see http://schemas.opengis.net/iso/19139/20070417/resources/codelist/gmxCodelists.xml"""
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
    attribution: t.Optional[str]  #: attribution (copyright) string
    contact: t.Optional[ContactConfig]  #: contact information
    dateCreated: t.Optional[t.Date]  #: publication date
    dateUpdated: t.Optional[t.Date]  #: modification date
    fees: t.Optional[str]
    image: t.Optional[t.Url]  #: image (logo) url

    insipreKeywords: t.Optional[t.List[MetaInspireKeyword]]
    insipreMandatoryKeyword: t.Optional[MetaInspireKeyword]
    inspireDegreeOfConformity: t.Optional[MetaInspireDegreeOfConformity]
    inspireResourceType: t.Optional[MetaInspireResourceType]
    inspireSpatialDataServiceType: t.Optional[MetaInspireSpatialDataServiceType]
    inspireTheme: t.Optional[str]  #: INSPIRE theme shortcut, like 'au'

    isoTopicCategory: t.Optional[MetaIsoTopicCategory]  #: ISO-19139 topic category
    isoQualityExplanation: t.Optional[str]
    isoQualityLineage: t.Optional[str]
    isoQualityPass: bool = False
    isoScope: t.Optional[MetaIsoScope]  #: ISO-19139 scope
    isoSpatialRepresentationType: t.Optional[MetaIsoSpatialRepresentationType]  #: ISO-19139 spatial type

    catalogUid: t.Optional[str]  #: catalog identifier

    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    links: t.List[LinkConfig] = []  #: additional links
    name: t.Optional[str]  #: object internal name
    serviceUrl: t.Optional[t.Url]  #: service url
    title: t.Optional[str]  #: object title
    url: t.Optional[t.Url]  #: metadata url
    urlType: t.Optional[str]  #: metadata url type like "ISO19115:2003"


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
    attribution: str
    contact: MetaContact
    dateCreated: t.Date
    dateUpdated: t.Date
    fees: str
    image: t.Url

    insipreMandatoryKeyword: str

    inspireResourceType: MetaInspireResourceType
    inspireSpatialDataServiceType: MetaInspireSpatialDataServiceType
    inspireTheme: str

    isoTopicCategory: MetaIsoTopicCategory
    isoQualityExplanation: str
    isoQualityLineage: str
    isoQualityPass: bool
    isoScope: MetaIsoScope
    isoSpatialRepresentationType: MetaIsoSpatialRepresentationType

    catalogUid: str

    keywords: t.List[str]
    language: str
    links: t.List[MetaLink]
    name: str
    serviceUrl: t.Url
    title: str
    url: t.Url
    urlType: str

    geographicExtent: t.Extent
    maxScale: int
    minScale: int
    proj: t.Projection


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
        meta.inspireThemeDefinition = inspire.theme_definition(meta.inspireTheme, meta.language)

    return meta


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

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
class MetaInspireTopicCategory(t.Enum):
    """Inspire topicCategory, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    farming = 'farming'
    biota = 'biota'
    boundaries = 'boundaries'
    climatologyMeteorologyAtmosphere = 'climatologyMeteorologyAtmosphere'
    economy = 'economy'
    elevation = 'elevation'
    environment = 'environment'
    geoscientificInformation = 'geoscientificInformation'
    health = 'health'
    imageryBaseMapsEarthCover = 'imageryBaseMapsEarthCover'
    intelligenceMilitary = 'intelligenceMilitary'
    inlandWaters = 'inlandWaters'
    location = 'location'
    oceans = 'oceans'
    planningCadastre = 'planningCadastre'
    society = 'society'
    structure = 'structure'
    transportation = 'transportation'
    utilitiesCommunication = 'utilitiesCommunication'


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
    """ISO-19139 MD_ScopeCode, see http://schemas.opengis.net/iso/19139/20070417/resources/codelist/ML_gmxCodelists.xml"""
    attribute = 'attribute'  #: Information applies to the attribute class
    attributeType = 'attributeType'  #: Information applies to the characteristic of a feature
    dataset = 'dataset'  #: Information applies to the dataset
    series = 'series'  #: Information applies to the series
    nonGeographicDataset = 'nonGeographicDataset'  #: Information applies to non-geographic data
    feature = 'feature'  #: Information applies to a feature
    featureType = 'featureType'  #: Information applies to a feature type
    propertyType = 'propertyType'  #: Information applies to a property type
    tile = 'tile'  #: Information applies to a tile, a spatial subset of geographic data


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
    dateCreated: t.Date = ''  #: publication date
    dateUpdated: t.Date = ''  #: modification date
    fees: t.Optional[str]
    image: t.Url = ''  #: image (logo) url
    images: dict = {}  #: further images

    insipreKeywords: t.Optional[t.List[MetaInspireKeyword]]
    insipreMandatoryKeyword: t.Optional[MetaInspireKeyword]
    inspireDegreeOfConformity: t.Optional[MetaInspireDegreeOfConformity]
    inspireResourceType: t.Optional[MetaInspireResourceType]
    inspireSpatialDataServiceType: t.Optional[MetaInspireSpatialDataServiceType]
    inspireTheme: t.Optional[str]  #: INSPIRE theme shortcut, like 'au'
    inspireTopicCategory: t.Optional[MetaInspireTopicCategory]

    isoTopicCategory: t.Optional[MetaIsoTopicCategory]  #: ISO-19139 topic category
    isoQualityExplanation: t.Optional[str]
    isoQualityLineage: t.Optional[str]
    isoQualityPass: bool = False
    isoScope: t.Optional[MetaIsoScope]  #: ISO-19139 scope
    isoSpatialRepresentationType: t.Optional[MetaIsoSpatialRepresentationType]  #: ISO-19139 spatial type
    isoUid: t.Optional[str]  #: ISO-19139 identifier

    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    links: t.List[LinkConfig] = []  #: additional links
    name: t.Optional[str]  #: object internal name
    serviceUrl: t.Url = ''  #: service url
    title: t.Optional[str]  #: object title
    url: t.Url = ''  #: metadata url


#:export
class MetaContact(t.Data):
    address = ''
    area = ''
    city = ''
    country = ''
    email = ''
    fax = ''
    organization = ''
    person = ''
    phone = ''
    position = ''
    zip = ''
    url = ''


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
    images: dict

    insipreMandatoryKeyword: str

    inspireResourceType: MetaInspireResourceType
    inspireSpatialDataServiceType: MetaInspireSpatialDataServiceType
    inspireTheme: str
    inspireTopicCategory: MetaInspireTopicCategory

    inspireThemeName: str
    inspireThemeDefinition: str

    isoTopicCategory: MetaIsoTopicCategory
    isoQualityExplanation: str
    isoQualityLineage: str
    isoQualityPass: bool
    isoScope: MetaIsoScope
    isoSpatialRepresentationType: MetaIsoSpatialRepresentationType
    isoUid: str

    keywords: t.List[str]
    language: str
    links: t.List[MetaLink]
    name: str
    serviceUrl: t.Url
    title: str
    url: t.Url

    geographicExtent: t.Optional[t.Extent]
    maxScale: t.Optional[int]
    minScale: t.Optional[int]
    proj: t.Projection


def from_config(m: t.Config) -> t.MetaData:
    if not m:
        return t.MetaData()

    meta = t.MetaData(m)

    meta.language = gws.get(meta, 'language') or 'en'
    meta.language3 = gws.tools.country.bibliographic_name(language=meta.language)

    meta.contact = MetaContact(gws.get(meta, 'contact') or {})

    if gws.get(meta, 'inspireTheme'):
        meta.inspireThemeName = inspire.theme_name(meta.inspireTheme, meta.language) or ''
        meta.inspireThemeDefinition = inspire.theme_definition(meta.inspireTheme, meta.language) or ''

    meta.links = [MetaLink(p) for p in (gws.get(meta, 'links') or [])]
    meta.keywords = gws.get(meta, 'keywords') or []

    return meta


def from_meta(m: t.MetaData) -> t.MetaData:
    return from_config(t.cast(t.Config, m))

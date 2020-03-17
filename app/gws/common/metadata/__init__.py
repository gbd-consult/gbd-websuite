"""Utilities to manipulate metadata"""

import gws
import gws.tools.country
import gws.types as t


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
    function: t.Optional[str]  #: ISO-19115 function, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#CI_OnLineFunctionCode


class InspireResourceType(t.Enum):
    """Inspire resourceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    dataset = 'dataset'
    series = 'series'
    service = 'service'


class InspireTopicCategory(t.Enum):
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


class InspireSpatialDataServiceType(t.Enum):
    """Inspire spatialDataServiceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    discovery = 'discovery'
    view = 'view'
    download = 'download'
    transformation = 'transformation'
    invoke = 'invoke'
    other = 'other'


class InspireDegreeOfConformity(t.Enum):
    """Inspire degreeOfConformity, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd"""
    conformant = 'conformant'
    notConformant = 'notConformant'
    notEvaluated = 'notEvaluated'


class InspireKeyword(t.Enum):
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

    insipreKeywords: t.Optional[t.List[InspireKeyword]]
    insipreMandatoryKeyword: t.Optional[InspireKeyword]
    inspireDegreeOfConformity: t.Optional[InspireDegreeOfConformity]
    inspireResourceType: t.Optional[InspireResourceType]
    inspireSpatialDataServiceType: t.Optional[InspireSpatialDataServiceType]
    inspireTheme: t.Optional[str]  #: INSPIRE theme shortcut, like 'au'
    inspireTopicCategory: t.Optional[InspireTopicCategory]

    isoCategory: t.Optional[str]  #: ISO-19115 category, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_TopicCategoryCode
    isoQualityExplanation: t.Optional[str]
    isoQualityLineage: t.Optional[str]
    isoQualityPass: bool = False
    isoScope: t.Optional[str]  #: ISO-19115 scope, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_ScopeCode
    isoSpatialType: t.Optional[str]  #: ISO-19115 spatial type, see http://standards.iso.org/ittf/PubliclyAvailableStandards/ISO_19139_Schemas/resources/codelist/ML_gmxCodelists.xml#MD_SpatialRepresentationTypeCode
    isoUid: t.Optional[str]  #: ISO-19115 identifier

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
    function: str


#:export
class MetaData(t.Data):
    abstract: str
    accessConstraints: str
    attribution: str
    contact: ContactConfig
    dateCreated: t.Date
    dateUpdated: t.Date
    fees: str
    image: t.Url
    images: dict

    insipreMandatoryKeyword: str

    inspireResourceType: InspireResourceType
    inspireSpatialDataServiceType: InspireSpatialDataServiceType
    inspireTheme: str
    inspireTopicCategory: InspireTopicCategory

    isoCategory: str
    isoQualityExplanation: str
    isoQualityLineage: str
    isoQualityPass: bool
    isoScope: str
    isoSpatialType: str
    isoUid: str

    keywords: t.List[str]
    language: str
    links: t.List[LinkConfig]
    name: str
    serviceUrl: t.Url
    title: str
    url: t.Url


def from_config(m) -> t.MetaData:
    if not m:
        return t.MetaData()
    meta = t.MetaData(m)
    if meta.get('language'):
        meta.language3 = gws.tools.country.bibliographic_name(language=m.language)
    if meta.get('contact'):
        meta.contact = MetaContact(meta.contact)
    meta.links = [MetaLink(p) for p in (meta.get('links') or [])]
    meta.keywords = meta.get('keywords') or []
    return meta

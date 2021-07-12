"""Utilities to manipulate metadata"""

import gws
import gws.types as t
import gws.lib.country
import gws.lib.date

from . import inspire

#: Inspire theme, see http://inspire.ec.europa.eu/theme/
InspireTheme = t.Literal[
    'ac',
    'ad',
    'af',
    'am',
    'au',
    'br',
    'bu',
    'cp',
    'ef',
    'el',
    'er',
    'ge',
    'gg',
    'gn',
    'hb',
    'hh',
    'hy',
    'lc',
    'lu',
    'mf',
    'mr',
    'nz',
    'of',
    'oi',
    'pd',
    'pf',
    'ps',
    'rs',
    'sd',
    'so',
    'sr',
    'su',
    'tn',
    'us',
]

#: Inspire resourceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd
InspireResourceType = t.Literal[
    'dataset',
    'series',
    'service',
]

#: Inspire spatialDataServiceType, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd
InspireSpatialDataServiceType = t.Literal[
    'discovery',
    'view',
    'download',
    'transformation',
    'invoke',
    'other',
]

#: Inspire spatialScope, see https://inspire.ec.europa.eu/metadata-codelist/SpatialScope
InspireSpatialScope = t.Literal[
    'national',
    'regional',
    'local',
    'global',
    'european',
]

#: Inspire degreeOfConformity, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd
InspireDegreeOfConformity = t.Literal[
    'conformant',
    'notConformant',
    'notEvaluated',
]

#: Inspire keyword, see http://inspire.ec.europa.eu/schemas/common/1.0/common.xsd
InspireMandatoryKeyword = t.Literal[
    'chainDefinitionService',
    'comEncodingService',
    'comGeographicCompressionService',
    'comGeographicFormatConversionService',
    'comMessagingService',
    'comRemoteFileAndExecutableManagement',
    'comService',
    'comTransferService',
    'humanCatalogueViewer',
    'humanChainDefinitionEditor',
    'humanFeatureGeneralizationEditor',
    'humanGeographicDataStructureViewer',
    'humanGeographicFeatureEditor',
    'humanGeographicSpreadsheetViewer',
    'humanGeographicSymbolEditor',
    'humanGeographicViewer',
    'humanInteractionService',
    'humanServiceEditor',
    'humanWorkflowEnactmentManager',
    'infoCatalogueService',
    'infoCoverageAccessService',
    'infoFeatureAccessService',
    'infoFeatureTypeService',
    'infoGazetteerService',
    'infoManagementService',
    'infoMapAccessService',
    'infoOrderHandlingService',
    'infoProductAccessService',
    'infoRegistryService',
    'infoSensorDescriptionService',
    'infoStandingOrderService',
    'metadataGeographicAnnotationService',
    'metadataProcessingService',
    'metadataStatisticalCalculationService',
    'spatialCoordinateConversionService',
    'spatialCoordinateTransformationService',
    'spatialCoverageVectorConversionService',
    'spatialDimensionMeasurementService',
    'spatialFeatureGeneralizationService',
    'spatialFeatureManipulationService',
    'spatialFeatureMatchingService',
    'spatialImageCoordinateConversionService',
    'spatialImageGeometryModelConversionService',
    'spatialOrthorectificationService',
    'spatialPositioningService',
    'spatialProcessingService',
    'spatialProximityAnalysisService',
    'spatialRectificationService',
    'spatialRouteDeterminationService',
    'spatialSamplingService',
    'spatialSensorGeometryModelAdjustmentService',
    'spatialSubsettingService',
    'spatialTilingChangeService',
    'subscriptionService',
    'taskManagementService',
    'temporalProcessingService',
    'temporalProximityAnalysisService',
    'temporalReferenceSystemTransformationService',
    'temporalSamplingService',
    'temporalSubsettingService',
    'thematicChangeDetectionService',
    'thematicClassificationService',
    'thematicFeatureGeneralizationService',
    'thematicGeocodingService',
    'thematicGeographicInformationExtractionService',
    'thematicGeoparsingService',
    'thematicGoparameterCalculationService',
    'thematicImageManipulationService',
    'thematicImageProcessingService',
    'thematicImageSynthesisService',
    'thematicImageUnderstandingService',
    'thematicMultibandImageManipulationService',
    'thematicObjectDetectionService',
    'thematicProcessingService',
    'thematicReducedResolutionGenerationService',
    'thematicSpatialCountingService',
    'thematicSubsettingService',
    'workflowEnactmentService',
]

#: ISO-19139 MD/MX_ScopeCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoScope = t.Literal[
    'attribute',
    'attributeType',
    'collectionHardware',
    'collectionSession',
    'dataset',
    'series',
    'nonGeographicDataset',
    'dimensionGroup',
    'feature',
    'featureType',
    'propertyType',
    'fieldSession',
    'software',
    'service',
    'model',
    'tile',
    'initiative',
    'stereomate',
    'sensor',
    'platformSeries',
    'sensorSeries',
    'productionSeries',
    'transferAggregate',
    'otherAggregate',
]

#: ISO-19139 MD_MaintenanceFrequencyCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoMaintenanceFrequencyCode = t.Literal[
    'continual',
    'daily',
    'weekly',
    'fortnightly',
    'monthly',
    'quarterly',
    'biannually',
    'annually',
    'asNeeded',
    'irregular',
    'notPlanned',
    'unknown',
]

#: ISO-19139 MD_SpatialRepresentationTypeCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoSpatialRepresentationType = t.Literal[
    'vector',
    'grid',
    'textTable',
    'tin',
    'stereoModel',
    'video',
]

#: ISO-19139 CI_OnLineFunctionCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoOnLineFunction = t.Literal[
    'download',
    'information',
    'offlineAccess',
    'order',
    'search',
]

#: ISO-19139 MD_TopicCategoryCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoTopicCategory = t.Literal[
    'farming',
    'biota',
    'boundaries',
    'climatologyMeteorologyAtmosphere',
    'economy',
    'elevation',
    'environment',
    'geoscientificInformation',
    'health',
    'imageryBaseMapsEarthCover',
    'intelligenceMilitary',
    'inlandWaters',
    'location',
    'oceans',
    'planningCadastre',
    'society',
    'structure',
    'transportation',
    'utilitiesCommunication',
]

#: ISO-19139 MD_RestrictionCode, see https://standards.iso.org/iso/19139/resources/gmxCodelists.xml
IsoRestrictionCode = t.Literal[
    'copyright',
    'patent',
    'patentPending',
    'trademark',
    'license',
    'intellectualPropertyRights',
    'restricted',
    'otherRestrictions',
]


class IsoQualityConformance(gws.Data):
    specificationTitle: str
    specificationDate: str
    explanation: str = ''
    qualityPass: bool


class IsoQualityLineage(gws.Data):
    statement: str
    source: str
    sourceScale: int


class ContactConfig(gws.Config):
    #: Contact metadata configuration

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
    url: t.Optional[gws.Url]


class LinkConfig(gws.Config):
    #: Object link configuration

    scheme: t.Optional[str]  #: link scheme
    url: gws.Url  #: link url
    formatName: t.Optional[str]  #: link format
    formatVersion: t.Optional[str]  #: link format version
    function: t.Optional[IsoOnLineFunction]  #: ISO-19115 function
    type: t.Optional[str]  #: metadata url type like "TC211"


class AuthorityConfig(gws.Config):
    name: t.Optional[str]
    url: t.Optional[gws.Url]
    identifier: t.Optional[str]


class InspireConfig(gws.Config):
    """INSPIRE metadata"""
    keywords: t.Optional[t.List[InspireMandatoryKeyword]]
    mandatoryKeyword: t.Optional[InspireMandatoryKeyword]
    degreeOfConformity: t.Optional[InspireDegreeOfConformity]
    resourceType: t.Optional[InspireResourceType]
    spatialDataServiceType: t.Optional[InspireSpatialDataServiceType]
    spatialScope: t.Optional[InspireSpatialScope]
    theme: t.Optional[InspireTheme]
    themeName: t.Optional[str]
    themeNameEn: t.Optional[str]


class IsoConfig(gws.Config):
    """ISO-19139 metadata"""
    maintenanceFrequencyCode: t.Optional[IsoMaintenanceFrequencyCode]
    scope: t.Optional[IsoScope]  #: ISO-19139 scope code
    scopeName: t.Optional[str]  #: ISO-19139 scope name
    spatialRepresentationType: t.Optional[IsoSpatialRepresentationType]  #: ISO-19139 spatial type
    topicCategory: t.Optional[IsoTopicCategory]  #: ISO-19139 topic category
    qualityConformance: t.Optional[t.List[IsoQualityConformance]]
    qualityLineage: t.Optional[IsoQualityLineage]
    restrictionCode: t.Optional[IsoRestrictionCode]


_list_props = ['keywords', 'links', ]
_obj_props = ['authority', 'contact', 'inspire', 'iso', 'metaLink', 'serviceLink', ]


class Data(gws.MetaData):
    abstract: t.Optional[str]  #: object abstract description
    accessConstraints: t.Optional[str]
    attribution: t.Optional[str]  #: attribution (copyright) string
    authority: t.Optional[AuthorityConfig]
    catalogCitationUid: t.Optional[str]  #: catalog citation identifier
    catalogUid: t.Optional[str]  #: catalog identifier
    contact: t.Optional[ContactConfig]  #: contact information
    dateBegin: t.Optional[gws.Date]  #: temporal extent begin
    dateCreated: t.Optional[gws.Date]  #: publication date
    dateEnd: t.Optional[gws.Date]  #: temporal extent end
    dateUpdated: t.Optional[gws.Date]  #: modification date
    fees: t.Optional[str]
    image: t.Optional[gws.Url]  #: image (logo) url
    inspire: t.Optional[InspireConfig]
    iso: t.Optional[IsoConfig]
    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    language3: t.Optional[str]  #: object language (bibliographic)
    license: t.Optional[str]
    links: t.List[LinkConfig] = []  #: additional links
    metaLink: t.Optional[LinkConfig]  #: metadata url
    name: t.Optional[str]  #: object internal name
    serviceLink: t.Optional[LinkConfig]  #: service url
    title: t.Optional[str]  #: object title


def from_dict(root: gws.RootObject, d: dict) -> Data:
    m = Data()

    for k, v in d.items():
        if '.' in k:
            k1, k2 = k.split('.', maxsplit=1)
            if not m.get(k1):
                m.set(k1, gws.Data())
            m.get(k1).set(v)
        else:
            m.set(k, v)

    if m.language:
        m.language3 = gws.lib.country.bibliographic_name(language=m.language)
    if m.inspire and m.inspire.theme:
        m.inspire.themeName = inspire.theme_name(m.inspire.theme, m.language)
        m.inspire.themeNameEn = inspire.theme_name(m.inspire.theme, 'en')

    return m


def merge(a, b) -> Data:
    ad = gws.as_dict(a)
    bd = gws.as_dict(b)

    m = gws.Data()

    for p in ad.keys() | bd.keys():
        if p in _list_props:
            m.set(p, gws.merge_lists(ad.get(p), bd.get(p)))
        elif p in _obj_props:
            m.set(p, gws.merge(ad.get(p), bd.get(p)))
        else:
            m.set(p, ad.get(p, bd.get(p)))

    return t.cast(Data, m)


class Config(Data):
    """metadata configuration"""
    pass


class Props(gws.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: t.List[str]
    language: str
    title: str


class Object(gws.Node, gws.IMeta):
    data: Data

    @property
    def props(self):
        return gws.Props(
            abstract=self.data.abstract or '',
            attribution=self.data.attribution or '',
            dateCreated=self.data.dateCreated,
            dateUpdated=self.data.dateUpdated,
            keywords=self.data.keywords or [],
            language=self.data.language or '',
            title=self.data.title or '',
        )

    def configure(self):
        self.data = Data(self.config)

    def extend(self, meta):
        self.data = merge(self.data, meta)

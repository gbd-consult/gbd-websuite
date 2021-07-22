"""Metadata structures and related utilities"""

import gws
import gws.lib.country
import gws.lib.date
import gws.types as t

from . import inspire


class Contact(gws.Data):
    """Contact metadata"""

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


class Link(gws.Data):
    """Link metadata"""

    scheme: t.Optional[str]  #: link scheme
    url: gws.Url  #: link url
    formatName: t.Optional[str]  #: link format
    formatVersion: t.Optional[str]  #: link format version
    function: t.Optional[str]  #: ISO-19115 online function code
    type: t.Optional[str]  #: metadata url type like "TC211"


class AuthorityConfig(gws.Data):
    name: t.Optional[str]
    url: t.Optional[gws.Url]
    identifier: t.Optional[str]


class InspireMeta(gws.Data):
    """INSPIRE metadata"""

    keywords: t.Optional[t.List[str]]  #: INSPIRE keywords
    mandatoryKeyword: t.Optional[str]  #: INSPIRE mandatory keyword
    degreeOfConformity: t.Optional[str]  #: INSPIRE degree of conformity
    resourceType: t.Optional[str]  #: INSPIRE resource type
    spatialDataServiceType: t.Optional[str]  #: INSPIRE spatial data service type
    spatialScope: t.Optional[str]  #: INSPIRE spatial scope
    theme: t.Optional[str]  #: INSPIRE theme, see http://inspire.ec.europa.eu/theme/
    themeName: t.Optional[str]  #: INSPIRE theme name, in the project language
    themeNameEn: t.Optional[str]  #: INSPIRE theme name, in English


class IsoQualityConformance(gws.Data):
    specificationTitle: str
    specificationDate: str
    explanation: str = ''
    qualityPass: bool


class IsoQualityLineage(gws.Data):
    statement: str
    source: str
    sourceScale: int


class IsoMeta(gws.Data):
    """ISO-19139 metadata"""

    maintenanceFrequencyCode: t.Optional[str]  #: ISO-19139 maintenance frequency code
    scope: t.Optional[str]  #: ISO-19139 scope code
    scopeName: t.Optional[str]  #: ISO-19139 scope name
    spatialRepresentationType: t.Optional[str]  #: ISO-19139 spatial type
    topicCategory: t.Optional[str]  #: ISO-19139 topic category
    qualityConformance: t.Optional[t.List[IsoQualityConformance]]  #: ISO-19139 quality conformance record
    qualityLineage: t.Optional[IsoQualityLineage]  #: ISO-19139 quality lineage record
    restrictionCode: t.Optional[str]  #: ISO-19139 restriction code


_list_props = ['keywords', 'links', ]
_obj_props = ['authority', 'contact', 'inspire', 'iso', 'metaLink', 'serviceLink', ]

# literal values for INSPIRE/ISO props
_literals = {
    'InspireMeta.theme': 'ac,ad,af,am,au,br,bu,cp,ef,el,er,ge,gg,gn,hb,hh,hy,lc,lu,mf,mr,nz,of,oi,pd,pf,ps,rs,sd,so,sr,su,tn,us',
    'InspireMeta.resourceType': 'dataset,series,service',
    'InspireMeta.spatialDataServiceType': 'discovery,view,download,transformation,invoke,other',
    'InspireMeta.spatialScope': 'national,regional,local,global,european',
    'InspireMeta.degreeOfConformity': 'conformant,notConformant,notEvaluated',
    'InspireMeta.mandatoryKeyword': 'chainDefinitionService,comEncodingService,comGeographicCompressionService,comGeographicFormatConversionService,comMessagingService,comRemoteFileAndExecutableManagement,comService,comTransferService,humanCatalogueViewer,humanChainDefinitionEditor,humanFeatureGeneralizationEditor,humanGeographicDataStructureViewer,humanGeographicFeatureEditor,humanGeographicSpreadsheetViewer,humanGeographicSymbolEditor,humanGeographicViewer,humanInteractionService,humanServiceEditor,humanWorkflowEnactmentManager,infoCatalogueService,infoCoverageAccessService,infoFeatureAccessService,infoFeatureTypeService,infoGazetteerService,infoManagementService,infoMapAccessService,infoOrderHandlingService,infoProductAccessService,infoRegistryService,infoSensorDescriptionService,infoStandingOrderService,metadataGeographicAnnotationService,metadataProcessingService,metadataStatisticalCalculationService,spatialCoordinateConversionService,spatialCoordinateTransformationService,spatialCoverageVectorConversionService,spatialDimensionMeasurementService,spatialFeatureGeneralizationService,spatialFeatureManipulationService,spatialFeatureMatchingService,spatialImageCoordinateConversionService,spatialImageGeometryModelConversionService,spatialOrthorectificationService,spatialPositioningService,spatialProcessingService,spatialProximityAnalysisService,spatialRectificationService,spatialRouteDeterminationService,spatialSamplingService,spatialSensorGeometryModelAdjustmentService,spatialSubsettingService,spatialTilingChangeService,subscriptionService,taskManagementService,temporalProcessingService,temporalProximityAnalysisService,temporalReferenceSystemTransformationService,temporalSamplingService,temporalSubsettingService,thematicChangeDetectionService,thematicClassificationService,thematicFeatureGeneralizationService,thematicGeocodingService,thematicGeographicInformationExtractionService,thematicGeoparsingService,thematicGoparameterCalculationService,thematicImageManipulationService,thematicImageProcessingService,thematicImageSynthesisService,thematicImageUnderstandingService,thematicMultibandImageManipulationService,thematicObjectDetectionService,thematicProcessingService,thematicReducedResolutionGenerationService,thematicSpatialCountingService,thematicSubsettingService,workflowEnactmentService',
    'IsoMeta.scope': 'attribute,attributeType,collectionHardware,collectionSession,dataset,series,nonGeographicDataset,dimensionGroup,feature,featureType,propertyType,fieldSession,software,service,model,tile,initiative,stereomate,sensor,platformSeries,sensorSeries,productionSeries,transferAggregate,otherAggregate',
    'IsoMeta.maintenanceFrequencyCode': 'continual,daily,weekly,fortnightly,monthly,quarterly,biannually,annually,asNeeded,irregular,notPlanned,unknown',
    'IsoMeta.spatialRepresentationType': 'vector,grid,textTable,tin,stereoModel,video',
    'Link.function': 'download,information,offlineAccess,order,search',
    'IsoMeta.topicCategory': 'farming,biota,boundaries,climatologyMeteorologyAtmosphere,economy,elevation,environment,geoscientificInformation,health,imageryBaseMapsEarthCover,intelligenceMilitary,inlandWaters,location,oceans,planningCadastre,society,structure,transportation,utilitiesCommunication',
    'IsoMeta.restrictionCode': 'copyright,patent,patentPending,trademark,license,intellectualPropertyRights,restricted,otherRestrictions',
}


class Values(gws.Data):
    abstract: t.Optional[str]  #: object abstract description
    accessConstraints: t.Optional[str]
    attribution: t.Optional[str]  #: attribution (copyright) string
    authority: t.Optional[AuthorityConfig]
    catalogCitationUid: t.Optional[str]  #: catalog citation identifier
    catalogUid: t.Optional[str]  #: catalog identifier
    contact: t.Optional[Contact]  #: contact information
    dateBegin: t.Optional[gws.Date]  #: temporal extent begin
    dateCreated: t.Optional[gws.Date]  #: publication date
    dateEnd: t.Optional[gws.Date]  #: temporal extent end
    dateUpdated: t.Optional[gws.Date]  #: modification date
    fees: t.Optional[str]
    image: t.Optional[gws.Url]  #: image (logo) url
    inspire: t.Optional[InspireMeta]
    iso: t.Optional[IsoMeta]
    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    language3: t.Optional[str]  #: object language (bibliographic)
    license: t.Optional[str]
    links: t.List[Link] = []  #: additional links
    metaLink: t.Optional[Link]  #: metadata url
    name: t.Optional[str]  #: object internal name
    serviceLink: t.Optional[Link]  #: service url
    title: t.Optional[str]  #: object title


def from_dict(d: dict) -> Values:
    m = Values()

    for k, v in d.items():
        if '.' in k:
            k1, k2 = k.split('.', maxsplit=1)
            if not m.get(k1):
                m.set(k1, gws.Data())
            m.get(k1).set(v)
        else:
            m.set(k, v)

    for p in _obj_props:
        m.set(p, gws.Data(m.get(p) or {}))

    if m.language:
        m.language3 = gws.lib.country.bibliographic_name(language=m.language)
    if m.inspire and m.inspire.theme:
        m.inspire.themeName = inspire.theme_name(m.inspire.theme, m.language)
        m.inspire.themeNameEn = inspire.theme_name(m.inspire.theme, 'en')

    return m


def merge(a, b) -> Values:
    ad = gws.as_dict(a)
    bd = gws.as_dict(b)

    m = Values()

    for p in ad.keys() | bd.keys():
        if p in _list_props:
            m.set(p, gws.merge_lists(ad.get(p), bd.get(p)))
        elif p in _obj_props:
            m.set(p, gws.merge(ad.get(p), bd.get(p)))
        else:
            m.set(p, ad.get(p, bd.get(p)))

    return m

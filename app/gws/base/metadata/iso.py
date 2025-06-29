"""ISO 19115 metadata."""

import gws


class MD_MaintenanceFrequencyCode(gws.Enum):
    """Frequency with which modifications and deletions are made to the data after it is first produced."""

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


class MD_RestrictionCode(gws.Enum):
    """Limitations placed upon the access or use of the data."""

    copyright = 'copyright'
    patent = 'patent'
    patentPending = 'patentPending'
    trademark = 'trademark'
    license = 'license'
    intellectualPropertyRights = 'intellectualPropertyRights'
    restricted = 'restricted'
    otherRestrictions = 'otherRestrictions'


class SV_ServiceFunction(gws.Enum):
    """Function performed by the service."""

    download = 'download'
    information = 'information'
    offlineAccess = 'offlineAccess'
    order = 'order'
    search = 'search'


class MD_ScopeCode(gws.Enum):
    """Class of information to which the referencing entity applies."""

    attribute = 'attribute'
    attributeType = 'attributeType'
    collectionHardware = 'collectionHardware'
    collectionSession = 'collectionSession'
    dataset = 'dataset'
    series = 'series'
    nonGeographicDataset = 'nonGeographicDataset'
    dimensionGroup = 'dimensionGroup'
    feature = 'feature'
    featureType = 'featureType'
    propertyType = 'propertyType'
    fieldSession = 'fieldSession'
    software = 'software'
    service = 'service'
    model = 'model'
    tile = 'tile'


class MD_SpatialRepresentationTypeCode(gws.Enum):
    """Method used to represent geographic information in the dataset."""

    vector = 'vector'
    grid = 'grid'
    textTable = 'textTable'
    tin = 'tin'
    stereoModel = 'stereoModel'
    video = 'video'


class MD_TopicCategoryCode(gws.Enum):
    """High-level geographic data thematic classification."""

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


class CI_RoleCode(gws.Enum):
    """Role of the responsible party."""

    resourceProvider = 'resourceProvider'
    custodian = 'custodian'
    owner = 'owner'
    user = 'user'
    distributor = 'distributor'
    originator = 'originator'
    pointOfContact = 'pointOfContact'
    principalInvestigator = 'principalInvestigator'
    processor = 'processor'
    publisher = 'publisher'
    author = 'author'
    sponsor = 'sponsor'
    contributor = 'contributor'
    rightsHolder = 'rightsHolder'
    editor = 'editor'


class CI_OnLineFunctionCode(gws.Enum):
    """Function performed by the online resource."""

    download = 'download'
    information = 'information'
    offlineAccess = 'offlineAccess'
    order = 'order'
    search = 'search'


class CI_PresentationFormCode(gws.Enum):
    """Format in which the resource is presented."""

    documentDigital = 'documentDigital'
    mapDigital = 'mapDigital'
    chartDigital = 'chartDigital'
    atlasDigital = 'atlasDigital'
    tableDigital = 'tableDigital'
    datasetDigital = 'datasetDigital'
    serviceDigital = 'serviceDigital'
    modelDigital = 'modelDigital'
    videoDigital = 'videoDigital'
    soundRecordingDigital = 'soundRecordingDigital'
    textTableDigital = 'textTableDigital'

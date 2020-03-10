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


class Config(t.Config):
    """Object metadata configuration"""

    abstract: t.Optional[str]  #: object abstract description
    accessConstraints: t.Optional[str]
    attribution: t.Optional[str]  #: attribution (copyright) string
    contact: t.Optional[ContactConfig]  #: contact information
    fees: t.Optional[str]
    image: t.Url = ''  #: image (logo) url
    images: dict = {}  #: further images
    isoCategory: t.Optional[str]  #: ISO-19115 category, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_TopicCategoryCode
    isoScope: t.Optional[str]  #: ISO-19115 scope, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_ScopeCode
    isoSpatialType: str = 'vector'  #: ISO-19115 spatial type, see http://standards.iso.org/ittf/PubliclyAvailableStandards/ISO_19139_Schemas/resources/codelist/ML_gmxCodelists.xml#MD_SpatialRepresentationTypeCode
    isoUid: t.Optional[str]  #: ISO-19115 identifier
    keywords: t.List[str] = []  #: keywords
    language: t.Optional[str]  #: object language
    links: t.List[LinkConfig] = []  #: additional links
    mandatoryKeyword: str = 'infoMapAccessService'
    modDate: t.Date = ''  #: modification date
    name: t.Optional[str]  #: object internal name
    inspireTheme: t.Optional[str] #: INSPIRE theme, like 'au'
    pubDate: t.Date = ''  #: publication date
    qualityExplanation: t.Optional[str]
    qualityLineage: t.Optional[str]
    qualityPass: bool = False
    resourceType: str = 'service'
    serviceUrl: t.Url = ''  #: service url
    spatialDataServiceType: t.Optional[str]
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
    contact: t.Optional[MetaContact]
    fees: str
    image: t.Url
    images: dict
    isoCategory: str
    isoScope: str
    isoSpatialType: str
    isoUid: str
    keywords: t.List[str]
    language: str
    language3: str
    links: t.List[MetaLink]
    mandatoryKeyword: str
    modDate: t.Date
    name: str
    inspireTheme: str
    pubDate: t.Date
    qualityExplanation: str
    qualityLineage: str
    qualityPass: bool
    resourceType: str
    serviceUrl: t.Url
    spatialDataServiceType: str
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

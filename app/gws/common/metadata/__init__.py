"""Utilities to manipulate metadata"""

import gws
import gws.tools.country
import gws.types as t


class ContactConfig(t.Config):
    """Contact metadata configuration"""

    address: str = ''
    area: str = ''
    city: str = ''
    country: str = ''
    email: str = ''
    fax: str = ''
    organization: str = ''
    person: str = ''
    phone: str = ''
    position: str = ''
    zip: str = ''
    url: t.Url = ''


class LinkConfig(t.Config):
    """Object link configuration"""

    scheme: str = ''  #: link scheme
    url: t.Url  #: link url
    function: str = ''  #: ISO-19115 function, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#CI_OnLineFunctionCode


class Config(t.Config):
    """Object metadata configuration"""

    abstract: str = ''  #: object abstract description
    attribution: str = ''  #: attribution (copyright) string
    keywords: t.List[str] = []  #: keywords
    language: str = ''  #: object language
    name: str = ''  #: object internal name
    title: str = ''  #: object title

    accessConstraints: str = ''
    fees: str = ''

    # uid: str = ''  #: ISO-19115 identifier
    # category: str = ''  #: ISO-19115 category, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_TopicCategoryCode
    # scope: str = ''  #: ISO-19115 scope, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_ScopeCode
    iso: dict = {}  #: ISO-19115 properties

    # theme: str = ''  #: INSPIRE theme shortcut, e.g. "au"
    inspire: dict = {}  #: INSPIRE  properties

    contact: t.Optional[ContactConfig]  #: contact information

    pubDate: t.Date = ''  #: publication date
    modDate: t.Date = ''  #: modification date

    image: t.Url = ''  #: image (logo) url
    images: dict = {}  #: further images

    url: t.Url = ''  #:  metadata url
    serviceUrl: t.Url = ''  #:  service url
    links: t.List[LinkConfig] = []  #: additional links


#:export
class MetaContact(t.Data):
    address: str = ''
    area: str = ''
    city: str = ''
    country: str = ''
    email: str = ''
    fax: str = ''
    organization: str = ''
    person: str = ''
    phone: str = ''
    position: str = ''
    zip: str = ''
    url: t.Url = ''


#:export
class MetaLink(t.Data):
    scheme: str = ''
    url: t.Url
    function: str = ''


#:export
class MetaData(t.Data):
    abstract: str = ''
    attribution: str = ''
    keywords: t.List[str] = []
    language: str = ''
    name: str = ''
    title: str = ''

    accessConstraints: str = ''
    fees: str = ''

    # uid: str = ''  #: ISO-19115 identifier
    # category: str = ''  #: ISO-19115 category, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_TopicCategoryCode
    # scope: str = ''  #: ISO-19115 scope, see https://geo-ide.noaa.gov/wiki/index.php?title=ISO_19115_and_19115-2_CodeList_Dictionaries#MD_ScopeCode
    iso: dict = {}

    # theme: str = ''  #: INSPIRE theme shortcut, e.g. "au"
    inspire: dict = {}

    contact: t.Optional[MetaContact]

    pubDate: t.Date = ''
    modDate: t.Date = ''

    image: t.Url = ''
    images: dict = {}

    url: t.Url = ''
    serviceUrl: t.Url = ''
    links: t.List[MetaLink] = []


def read(m: t.MetaData) -> t.MetaData:
    if not m:
        return t.MetaData()
    if m.get('language'):
        m.language3 = gws.tools.country.bibliographic_name(language=m.language)
    return m

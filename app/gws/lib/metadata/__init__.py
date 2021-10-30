"""Metadata structures and related utilities"""

import gws
import gws.lib.country
import gws.lib.date
import gws.types as t

from . import inspire


class Link(gws.Data):
    """Link metadata"""

    scheme: t.Optional[str]  #: link scheme
    url: gws.Url  #: link url
    formatName: t.Optional[str]  #: link format
    formatVersion: t.Optional[str]  #: link format version
    function: t.Optional[str]  #: ISO-19115 online function code
    type: t.Optional[str]  #: metadata url type like "TC211"


class Values(gws.Data):
    abstract: t.Optional[str]  #: object abstract description
    accessConstraints: t.Optional[str]
    attribution: t.Optional[str]  #: attribution (copyright) string

    authorityIdentifier: t.Optional[str]
    authorityName: t.Optional[str]
    authorityUrl: t.Optional[gws.Url]

    catalogCitationUid: t.Optional[str]  #: catalog citation identifier
    catalogUid: t.Optional[str]  #: catalog identifier

    contactAddress: t.Optional[str]
    contactArea: t.Optional[str]
    contactCity: t.Optional[str]
    contactCountry: t.Optional[str]
    contactEmail: t.Optional[str]
    contactFax: t.Optional[str]
    contactOrganization: t.Optional[str]
    contactPerson: t.Optional[str]
    contactPhone: t.Optional[str]
    contactPosition: t.Optional[str]
    contactZip: t.Optional[str]
    contactUrl: t.Optional[gws.Url]

    dateBegin: t.Optional[gws.Date]  #: temporal extent begin
    dateCreated: t.Optional[gws.Date]  #: publication date
    dateEnd: t.Optional[gws.Date]  #: temporal extent end
    dateUpdated: t.Optional[gws.Date]  #: modification date

    fees: t.Optional[str]
    image: t.Optional[gws.Url]  #: image (logo) url

    inspireKeywords: t.List[str] = []  #: INSPIRE keywords
    inspireMandatoryKeyword: t.Optional[str]  #: INSPIRE mandatory keyword
    inspireDegreeOfConformity: t.Optional[str]  #: INSPIRE degree of conformity
    inspireResourceType: t.Optional[str]  #: INSPIRE resource type
    inspireSpatialDataServiceType: t.Optional[str]  #: INSPIRE spatial data service type
    inspireSpatialScope: t.Optional[str]  #: INSPIRE spatial scope
    inspireSpatialScopeName: t.Optional[str]  #: INSPIRE spatial scope localized name
    inspireTheme: t.Optional[str]  #: INSPIRE theme, see http://inspire.ec.europa.eu/theme/
    inspireThemeName: t.Optional[str]  #: INSPIRE theme name, in the project language
    inspireThemeNameEn: t.Optional[str]  #: INSPIRE theme name, in English

    isoMaintenanceFrequencyCode: t.Optional[str]  #: ISO-19139 maintenance frequency code
    isoQualityConformanceExplanation: t.Optional[str]
    isoQualityConformanceQualityPass: t.Optional[bool]
    isoQualityConformanceSpecificationDate: t.Optional[str]
    isoQualityConformanceSpecificationTitle: t.Optional[str]
    isoQualityLineageSource: t.Optional[str]
    isoQualityLineageSourceScale: t.Optional[int]
    isoQualityLineageStatement: t.Optional[str]
    isoRestrictionCode: t.Optional[str]  #: ISO-19139 restriction code
    isoScope: t.Optional[str]  #: ISO-19139 scope code
    isoScopeName: t.Optional[str]  #: ISO-19139 scope name
    isoSpatialRepresentationType: t.Optional[str]  #: ISO-19139 spatial type
    isoTopicCategory: t.Optional[str]  #: ISO-19139 topic category
    isoSpatialResolution: t.Optional[str]  #: ISO-19139 spatial resolution

    keywords: t.List[str] = []  #: keywords
    language3: t.Optional[str]  #: object language (bibliographic)
    language: t.Optional[str]  #: object language
    languageName: t.Optional[str]  #: localized language name
    license: t.Optional[str]
    name: t.Optional[str]  #: object internal name
    title: t.Optional[str]  #: object title

    metaLinks: t.List[Link] = []  #: metadata links
    extraLinks: t.List[Link] = []  #: additional links


##

class Props(gws.Props):
    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: t.List[str]
    language: str
    title: str


class ExtendOption(t.Enum):
    app = 'app'  #: substutute missing metadata from the project or application config
    source = 'source'  #: substutute missing metadata from the source


class Config(Values):
    """Metadata configuration"""

    extend: t.Optional[ExtendOption]


##

def from_dict(d: dict) -> 'Metadata':
    return Metadata(d)


def from_args(**kwargs) -> 'Metadata':
    return from_dict(kwargs)


def from_config(cfg: gws.Config) -> 'Metadata':
    return from_dict(gws.to_dict(cfg))


def from_props(props: gws.Props) -> 'Metadata':
    return from_dict(gws.to_dict(props))


##


_LIST_PROPS = {'metaLinks', 'extraLinks'}
_EXT_LIST_PROPS = {'keywords', 'inspireKeywords'}
_NO_EXT_PROPS = {'authorityIdentifier', 'catalogUid'}

class Metadata(gws.Object, gws.IMetadata):
    values: Values

    def __init__(self, d):
        super().__init__()
        self._update(d)

    def props_for(self, user):
        return Props(
            abstract=self.values.abstract or '',
            attribution=self.values.attribution or '',
            dateCreated=self.values.dateCreated,
            dateUpdated=self.values.dateUpdated,
            keywords=self.values.keywords or [],
            language=self.values.language or '',
            title=self.values.title or '',
        )

    def extend(self, *others):
        d = gws.to_dict(self.values)

        for o in others:
            if not o:
                continue

            if isinstance(o, Metadata):
                o = o.values

            for k, new in gws.to_dict(o).items():
                if k in _NO_EXT_PROPS or gws.is_empty(new):
                    continue
                old = d.get(k)
                if old is None:
                    d[k] = new
                elif k in _EXT_LIST_PROPS:
                    old.extend(new)

        self._update(d)
        return self

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, val):
        d = gws.to_dict(self.values)
        d[key] = val
        self._update(d)
        return self

    def _update(self, d):
        for k in _EXT_LIST_PROPS:
            d[k] = sorted(set(d.get(k, [])))
        for k in _LIST_PROPS:
            d[k] = d.get(k, [])

        if d.get('language'):
            d['language3'] = gws.lib.country.bibliographic_name(language=d['language'])

        if d.get('inspireTheme'):
            d['inspireThemeName'] = inspire.theme_name(d['inspireTheme'], d.get('language'))
            d['inspireThemeNameEn'] = inspire.theme_name(d['inspireTheme'], 'en')

        self.values = Values(d)

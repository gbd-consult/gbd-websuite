"""Metadata structures and related utilities"""

import gws
import gws.lib.intl
import gws.lib.metadata.inspire
import gws.types as t


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


# NB this must be kept in sync with gws.MetadataValues

class MetadataLinkConfig(gws.Data):
    """Link metadata"""

    scheme: t.Optional[str]  #: link scheme
    url: gws.Url  #: link url
    formatName: t.Optional[str]  #: link format
    formatVersion: t.Optional[str]  #: link format version
    function: t.Optional[str]  #: ISO-19115 online function code
    type: t.Optional[str]  #: metadata url type like "TC211"


class Config(gws.Config):
    """Metadata configuration"""

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
    contactProviderName: t.Optional[str]
    contactProviderSite: t.Optional[str]
    contactRole: t.Optional[str]  #: https://standards.iso.org/iso/19139/resources/gmxCodet.Lists.xml#CI_RoleCode
    contactUrl: t.Optional[gws.Url]
    contactZip: t.Optional[str]

    dateBegin: t.Optional[gws.Date]  #: temporal extent begin
    dateCreated: t.Optional[gws.Date]  #: publication date
    dateEnd: t.Optional[gws.Date]  #: temporal extent end
    dateUpdated: t.Optional[gws.Date]  #: modification date

    fees: t.Optional[str]
    image: t.Optional[gws.Url]  #: image (logo) url

    inspireKeywords: t.Optional[t.List[str]]  #: INSPIRE keywords
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
    isoQualityConformanceSpecificationDate: t.Optional[gws.Date]
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

    keywords: t.Optional[t.List[str]]  #: keywords

    language3: t.Optional[str]  #: object language (bibliographic)
    language: t.Optional[str]  #: object language
    languageName: t.Optional[str]  #: localized language name

    license: t.Optional[str]
    name: t.Optional[str]  #: object internal name
    title: t.Optional[str]  #: object title

    metaLinks: t.Optional[t.List[MetadataLinkConfig]]  #: metadata links
    extraLinks: t.Optional[t.List[MetadataLinkConfig]]  #: additional links

    url: t.Optional[gws.Url]


##

def from_dict(d: dict) -> 'Object':
    return Object(d)


def from_args(**kwargs) -> 'Object':
    return from_dict(kwargs)


def from_config(cfg: gws.Config) -> 'Object':
    return from_dict(gws.to_dict(cfg))


def from_props(props: gws.Props) -> 'Object':
    return from_dict(gws.to_dict(props))


##


_LIST_PROPS = {'metaLinks', 'extraLinks', 'keywords', 'inspireKeywords'}
_EXTENDABLE_LIST_PROPS = {'keywords', 'inspireKeywords'}
_NON_EXTENDABLE_PROPS = {'authorityIdentifier', 'catalogUid'}


class Object(gws.Object, gws.IMetadata):
    def __init__(self, d):
        self.values = gws.MetadataValues(d)
        self._ensure_consistency()

    def props(self, user):
        return gws.Data(
            abstract=self.values.abstract or '',
            attribution=self.values.attribution or '',
            dateCreated=self.values.dateCreated,
            dateUpdated=self.values.dateUpdated,
            keywords=self.values.keywords or [],
            language=self.values.language or '',
            title=self.values.title or '',
        )

    def extend(self, *others):
        vs = vars(self.values)

        for other in others:
            if not other:
                continue
            if isinstance(other, Object):
                other = other.values

            for k, new in gws.to_dict(other).items():
                if k in _NON_EXTENDABLE_PROPS or gws.is_empty(new):
                    continue
                old = vs.get(k)
                if old is None:
                    vs[k] = new
                elif k in _EXTENDABLE_LIST_PROPS:
                    vs[k] = sorted(set(old + new))

        self._ensure_consistency()
        return self

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, val):
        vars(self.values)[key] = val
        self._ensure_consistency()
        return self

    def _ensure_consistency(self):
        vs = vars(self.values)

        for k in _LIST_PROPS:
            vs[k] = vs.get(k) or []

        if vs.get('language'):
            vs['language3'] = gws.lib.intl.bibliographic_name(language=vs['language'])

        if vs.get('inspireTheme'):
            vs['inspireThemeName'] = gws.lib.metadata.inspire.theme_name(vs['inspireTheme'], vs.get('language'))
            vs['inspireThemeNameEn'] = gws.lib.metadata.inspire.theme_name(vs['inspireTheme'], 'en')

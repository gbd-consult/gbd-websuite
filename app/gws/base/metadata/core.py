from typing import Optional
import gws
import gws.lib.intl
import gws.lib.datetimex as dtx

from . import inspire, iso


class LinkConfig(gws.Config):
    """Link to a metadata document or another resource related to the object."""

    about: Optional[str]
    """Aspect of the object the link describes."""
    description: Optional[str]
    """Description of the linked resource."""
    format: Optional[str]
    """Format of the linked resource."""
    formatVersion: Optional[str]
    """Version of the format of the linked resource."""
    function: Optional[str]
    """Function of the link."""
    mimeType: Optional[gws.MimeType]
    """MIME type of the linked resource."""
    scheme: Optional[str]
    """Link scheme or protocol."""
    title: Optional[str]
    """Link title."""
    type: Optional[str]
    url: Optional[str]
    """Link URL."""


class Config(gws.Config):
    """Metadata of an object, used in the client and in OWS services."""

    name: Optional[str]
    """Object name."""
    title: Optional[str]
    """Title of the object."""

    abstract: Optional[str]
    """Abstract of the object, a brief description."""
    accessConstraints: Optional[str]
    """Access constraints, written to AccessConstraints in service capabilities."""
    accessConstraintsType: Optional[str]
    """Access constraint type for the object."""
    attribution: Optional[str]
    """Attribution text, shown in the client and as the WMS Attribution title."""
    attributionUrl: Optional[str]
    """URL of the attribution, written to the WMS Attribution online resource."""
    dateCreated: Optional[gws.DateStr]
    """Creation date of the object."""
    dateUpdated: Optional[gws.DateStr]
    """Last update date of the object."""
    fees: Optional[str]
    """Fees for using the object, written to Fees in service capabilities."""
    image: Optional[str]
    """Image URL or path associated with the object."""
    keywords: Optional[list[str]]
    """Keywords, optionally prefixed with a vocabulary, e.g. 'gemet:river'."""
    license: Optional[str]
    """License text, written to the legal constraints in CSW records."""
    licenseUrl: Optional[gws.Url]
    """License URL."""

    contactAddress: Optional[str]
    """Street address of the contact."""
    contactAddressType: Optional[str]
    """Type of contact address, such as 'postal' or 'email'."""
    contactArea: Optional[str]
    """Administrative area, state or province of the contact address."""
    contactCity: Optional[str]
    """Contact city."""
    contactCountry: Optional[str]
    """Contact country."""
    contactEmail: Optional[str]
    """Contact email address."""
    contactFax: Optional[str]
    """Contact fax number."""
    contactOrganization: Optional[str]
    """Contact organization or institution."""
    contactPerson: Optional[str]
    """Contact person name."""
    contactPhone: Optional[str]
    """Contact phone number."""
    contactPosition: Optional[str]
    """Contact position or job title."""
    contactProviderName: Optional[str]
    """Name of the service provider, written to ServiceProvider/ProviderName."""
    contactProviderSite: Optional[str]
    """Website of the service provider, written to ServiceProvider/ProviderSite."""
    contactRole: Optional[iso.CI_RoleCode]
    """Role of the contact."""
    contactUrl: Optional[str]
    """Website of the contact, written to the contact OnlineResource."""
    contactZip: Optional[str]
    """Contact postal code."""

    authorityIdentifier: Optional[str]
    """Layer identifier issued by the authority, written to the WMS Identifier element."""
    authorityName: Optional[str]
    """Name of the authority that issues layer identifiers."""
    authorityUrl: Optional[str]
    """URL of the authority that issues layer identifiers."""

    metaLinks: Optional[list[LinkConfig]]
    """Links to metadata documents."""
    serviceMetadataURL: Optional[str]
    """URL of the service metadata document."""

    catalogCitationUid: Optional[str]
    """Identifier of the resource, written to the CI_Citation identifier in CSW records."""
    catalogUid: Optional[str]
    """Identifier of the metadata record."""

    language: Optional[str]
    """Language of the object as an ISO 639-1 code."""

    parentIdentifier: Optional[str]
    """Identifier of the parent metadata record."""
    wgsExtent: Optional[gws.Extent]
    """Geographic extent in WGS84, written to EX_Extent in CSW records."""
    crs: Optional[gws.CrsName]
    """Reference system of the data, written to MD_ReferenceSystem in CSW records."""
    temporalBegin: Optional[gws.DateStr]
    """Start of the temporal extent of the data."""
    temporalEnd: Optional[gws.DateStr]
    """End of the temporal extent of the data."""

    inspireMandatoryKeyword: Optional[inspire.IM_MandatoryKeyword]
    """INSPIRE service type, added to the keywords as the ISO serviceType keyword."""
    inspireDegreeOfConformity: Optional[inspire.IM_DegreeOfConformity]
    """Degree of conformity with the INSPIRE implementing rules."""
    inspireResourceType: Optional[inspire.IM_ResourceType]
    """INSPIRE resource type."""
    inspireSpatialDataServiceType: Optional[inspire.IM_SpatialDataServiceType]
    """INSPIRE spatial data service type."""
    inspireSpatialScope: Optional[inspire.IM_SpatialScope]
    """INSPIRE spatial scope."""
    inspireSpatialScopeName: Optional[str]
    """Display name of the INSPIRE spatial scope."""
    inspireTheme: Optional[inspire.IM_Theme]
    """INSPIRE data theme, added to the keywords as a GEMET INSPIRE theme."""

    isoMaintenanceFrequencyCode: Optional[iso.MD_MaintenanceFrequencyCode]
    """How often the data is updated."""
    isoQualityConformanceExplanation: Optional[str]
    """Explanation of the conformance result."""
    isoQualityConformanceQualityPass: Optional[bool]
    """The data passes the conformance test."""
    isoQualityConformanceSpecificationDate: Optional[str]
    """Publication date of the specification the conformance is tested against."""
    isoQualityConformanceSpecificationTitle: Optional[str]
    """Title of the specification the conformance is tested against."""
    isoQualityLineageSource: Optional[str]
    """Description of the source data."""
    isoQualityLineageSourceScale: Optional[int]
    """Scale denominator of the source data."""
    isoQualityLineageStatement: Optional[str]
    """Statement on the lineage of the data."""
    isoRestrictionCode: Optional[iso.MD_RestrictionCode]
    """Restrictions on access or use of the data."""
    isoServiceFunction: Optional[iso.SV_ServiceFunction]
    """ISO service function."""
    isoScope: Optional[iso.MD_ScopeCode]
    """Scope of the metadata."""
    isoScopeName: Optional[str]
    """Name of the scope, written to hierarchyLevelName in CSW records."""
    isoSpatialRepresentationType: Optional[iso.MD_SpatialRepresentationTypeCode]
    """How the data is represented spatially."""
    isoTopicCategories: Optional[list[iso.MD_TopicCategoryCode]]
    """ISO 19115 topic categories, added to the keywords."""
    isoSpatialResolution: Optional[int]
    """Spatial resolution as a scale denominator."""


##

_KEYWORD_CODE_SPACES = {
    'iso': ['ISOTC211/19115', 'http://www.isotc211.org/2005/resources/Codelist/gmxCodelists.xml#MD_KeywordTypeCode'],
    'gemet': ['GEMET', 'http://www.eionet.europa.eu/gemet/2004/06/gemet-version.rdf'],
    'inspire_themes': ['GEMET - INSPIRE themes', 'http://inspire.ec.europa.eu/theme'],
    'gcmd': ['gcmd', 'http://gcmd.nasa.gov/Resources/valids/locations.html'],
}


class KeywordGroup(gws.Data):
    codeSpace: str
    """Code space for the keyword group, e.g. 'iso', 'gemet', 'inspire', 'gcmd'."""
    typeName: str
    """Type name for the keyword group, e.g. 'isoTopicCategories', 'keywords'."""
    keywords: list[str]
    """List of keywords in the group."""


def keyword_groups(md: gws.Metadata) -> list[KeywordGroup]:
    d = {}

    def add(kw):
        p = kw.split(':')
        if len(p) == 1:
            return add2('', '', kw)

    def add2(code_space, type_name, kw):
        if code_space.lower() in _KEYWORD_CODE_SPACES:
            code_space = _KEYWORD_CODE_SPACES[code_space.lower()][0]
        key = (code_space, type_name)
        if key not in d:
            d[key] = KeywordGroup(codeSpace=code_space, typeName=type_name, keywords=[])
        d[key].keywords.append(kw)

    if md.keywords:
        for kw in md.keywords:
            add(kw)
    if md.inspireTheme:
        add2('inspire_themes', 'theme', md.inspireTheme)
    if md.isoTopicCategories:
        for cat in md.isoTopicCategories:
            add2('iso', 'isoTopicCategory', cat)
    if md.inspireMandatoryKeyword:
        add2('iso', 'serviceType', md.inspireMandatoryKeyword)

    return list(d.values())


class Props(gws.Props):
    """Represents metadata properties."""

    abstract: str
    attribution: str
    dateCreated: str
    dateUpdated: str
    keywords: list[str]
    language: str
    title: str


def new() -> gws.Metadata:
    """Create a new Metadata object with default values."""

    return _new()


def from_dict(d: dict) -> gws.Metadata:
    """Create a Metadata object from a dictionary.

    Args:
        d: Dictionary containing metadata information.
    """

    return _update(_new(), d)


def from_args(*args, **kwargs) -> gws.Metadata:
    """Create a Metadata object from arguments (dicts or other Metadata objects)."""

    return _update(_new(), *args, **kwargs)


def from_config(c: gws.Config) -> gws.Metadata:
    """Create a Metadata object from a configuration.

    Args:
        c: Configuration object.
    """

    return _update(_new(), c)


def from_props(p: gws.Props) -> gws.Metadata:
    """Create a Metadata object from properties.

    Args:
        p: Properties object.
    """

    return _update(_new(), p)


def update(md: gws.Metadata, *args, **kwargs) -> gws.Metadata:
    """Update a Metadata object from arguments (dicts or other Metadata objects)."""

    _update(md, *args, **kwargs)
    return md


def normalize(md: gws.Metadata) -> gws.Metadata:
    """Normalize a Metadata object (e.g. fix dates and language codes)."""

    nor = _new()
    _update(nor, md)
    return nor


def props(md: gws.Metadata) -> gws.Props:
    """Properties of a Metadata object."""

    dc = dtx.parse(md.dateCreated)
    du = dtx.parse(md.dateUpdated)

    return gws.Props(
        abstract=md.abstract or '',
        attribution=md.attribution or '',
        dateCreated=dtx.to_iso_date_string(dc) if dc else '',
        dateUpdated=dtx.to_iso_date_string(du) if du else '',
        keywords=sorted(md.keywords or []),
        language=md.language or '',
        title=md.title or '',
    )


##


def _new() -> gws.Metadata:
    md = gws.Metadata()
    for key, fn in _UPDATE_FNS.items():
        fn(md, key, None)
    return md


def _update(md: gws.Metadata, *args, **kwargs):
    def add(a):
        for key, val in a.items():
            fn = _UPDATE_FNS.get(key)
            if fn:
                fn(md, key, val)
            elif val is not None:
                setattr(md, key, val)

    for a in args:
        if not a:
            continue
        if isinstance(a, gws.Data):
            a = gws.u.to_dict(a)
        add(a)

    add(kwargs)
    _fix_language(md)

    return md


def _update_set(md: gws.Metadata, key, val):
    s = set(getattr(md, key, None) or [])
    s.update(val or [])
    setattr(md, key, sorted(s))


def _update_list(md: gws.Metadata, key, val):
    setattr(md, key, val or [])


def _update_datetime(md: gws.Metadata, key, val):
    if val:
        dt = dtx.parse(val)
        if dt:
            setattr(md, key, dt)


def _fix_language(md: gws.Metadata):
    lang = md.language or 'en'

    md.language3 = gws.lib.intl.locale(lang).language3
    md.languageBib = gws.lib.intl.locale(lang).languageBib

    if md.inspireTheme:
        md.inspireThemeNameLocal = inspire.theme_name(md.inspireTheme, md.language) or ''
        md.inspireThemeNameEn = inspire.theme_name(md.inspireTheme, 'en') or ''


_UPDATE_FNS = dict(
    keywords=_update_set,
    isoTopicCategories=_update_set,
    metaLinks=_update_list,
    dateCreated=_update_datetime,
    dateUpdated=_update_datetime,
    temporalBegin=_update_datetime,
    temporalEnd=_update_datetime,
)

from typing import Optional
import gws
import gws.lib.intl

from . import inspire, iso


class LinkConfig(gws.Config):
    """Metadata link."""

    about: Optional[str]
    description: Optional[str]
    format: Optional[str]
    formatVersion: Optional[str]
    function: Optional[str]
    mimeType: Optional[gws.MimeType]
    scheme: Optional[str]
    title: Optional[str]
    type: Optional[str]
    url: Optional[str]


class Config(gws.Config):
    """Metadata configuration. (added in 8.2)"""

    name: Optional[str]
    """Object name."""
    title: Optional[str]
    """Object title."""

    abstract: Optional[str]
    """Object abstract, a brief description of the object."""
    accessConstraints: Optional[str]
    """Access constraint for the object."""
    accessConstraintsType: Optional[str]
    """Access constraint type for the object."""
    attribution: Optional[str]
    """Attribution information for the object. (changed in 8.2)"""
    attributionUrl: Optional[str]
    """Attribution URL for the object. (added in 8.2)"""
    dateCreated: Optional[gws.DateStr]
    """Object creation date."""
    dateUpdated: Optional[gws.DateStr]
    """Object last update date."""
    fees: Optional[str]
    """Fees associated with accessing or using the object."""
    image: Optional[str]
    """Image URL or path associated with the object."""
    keywords: Optional[list[str]]
    """Keywords, optionally prefixed with a vocabulary, e.g. 'gemet:river'."""
    license: Optional[str]
    """License information for the object."""
    licenseUrl: Optional[gws.Url]
    """License URL."""

    contactAddress: Optional[str]
    """Contact address for the object."""
    contactAddressType: Optional[str]
    """Type of contact address, such as 'postal' or 'email'."""
    contactArea: Optional[str]
    """Contact area or state."""
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
    """Name of the provider of the contact information."""
    contactProviderSite: Optional[str]
    """Website of the provider of the contact information."""
    contactRole: Optional[iso.CI_RoleCode]
    """Role of the contact person, such as 'pointOfContact' or 'author'."""
    contactUrl: Optional[str]
    """URL for additional contact information."""
    contactZip: Optional[str]
    """Contact postal code."""

    authorityIdentifier: Optional[str]
    """Identifier (WMS)"""
    authorityName: Optional[str]
    """AuthorityURL name (WMS)"""
    authorityUrl: Optional[str]
    """AuthorityURL (WMS)"""

    metaLinks: Optional[list[LinkConfig]]
    """MetadataURL (WMS, WFS) or metadata links (CSW)."""
    serviceMetadataURL: Optional[str]
    """Service metadata URL (WMTS)."""

    catalogCitationUid: Optional[str]
    """CI_Citation.Identifier (CSW)."""
    catalogUid: Optional[str]
    """MD_Metadata.Identifier (CSW)."""

    language: Optional[str]
    """Language code (ISO 639-1)."""

    parentIdentifier: Optional[str]
    """MD_Metadata.parentIdentifier (ISO)."""
    wgsExtent: Optional[gws.Extent]
    """EX_Extent (ISO)."""
    crs: Optional[gws.CrsName]
    """MD_ReferenceSystem (ISO)."""
    temporalBegin: Optional[gws.DateStr]
    """EX_TemporalExtent (ISO)."""
    temporalEnd: Optional[gws.DateStr]
    """EX_TemporalExtent (ISO)."""

    inspireMandatoryKeyword: Optional[inspire.IM_MandatoryKeyword]
    inspireDegreeOfConformity: Optional[inspire.IM_DegreeOfConformity]
    inspireResourceType: Optional[inspire.IM_ResourceType]
    inspireSpatialDataServiceType: Optional[inspire.IM_SpatialDataServiceType]
    inspireSpatialScope: Optional[inspire.IM_SpatialScope]
    inspireSpatialScopeName: Optional[str]
    inspireTheme: Optional[inspire.IM_Theme]

    isoMaintenanceFrequencyCode: Optional[iso.MD_MaintenanceFrequencyCode]
    isoQualityConformanceExplanation: Optional[str]
    isoQualityConformanceQualityPass: Optional[bool]
    isoQualityConformanceSpecificationDate: Optional[str]
    isoQualityConformanceSpecificationTitle: Optional[str]
    isoQualityLineageSource: Optional[str]
    isoQualityLineageSourceScale: Optional[int]
    isoQualityLineageStatement: Optional[str]
    isoRestrictionCode: Optional[iso.MD_RestrictionCode]
    isoServiceFunction: Optional[iso.SV_ServiceFunction]
    isoScope: Optional[iso.MD_ScopeCode]
    isoScopeName: Optional[str]
    isoSpatialRepresentationType: Optional[iso.MD_SpatialRepresentationTypeCode]
    isoTopicCategories: Optional[list[iso.MD_TopicCategoryCode]]
    isoSpatialResolution: Optional[str]


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


def props(md: gws.Metadata) -> gws.Props:
    """Properties of a Metadata object."""

    return gws.Props(
        abstract=md.abstract or '',
        attribution=md.attribution or '',
        dateCreated=md.dateCreated,
        dateUpdated=md.dateUpdated,
        keywords=sorted(md.keywords or []),
        language=md.language or '',
        title=md.title or '',
    )


##


def _new() -> gws.Metadata:
    md = gws.Metadata()
    for key in vars(_Updaters):
        if key.startswith('_'):
            continue
        fn = getattr(_Updaters, key)
        fn(md, key, None)
    return md


def _update(md: gws.Metadata, *args, **kwargs):
    d = {}
    for arg in list(args):
        if not arg:
            continue
        if isinstance(arg, gws.Data):
            arg = gws.u.to_dict(arg)
        d.update(arg)
    d.update(kwargs)
    
    for key, val in d.items():
        fn = getattr(_Updaters, key, None)
        if fn:
            fn(md, key, val)
        elif val is not None:
            setattr(md, key, val)

    if 'inspireTheme' in d:
        _update_inspire_theme(md, '', d['inspireTheme'])

    return md


def _update_set(md: gws.Metadata, key, val):
    s = set(getattr(md, key, None) or [])
    s.update(val or [])
    setattr(md, key, sorted(s))


def _update_list(md: gws.Metadata, key, val):
    setattr(md, key, val or [])


def _update_language(md: gws.Metadata, key, val):
    if val:
        md.language = val
        md.language3 = gws.lib.intl.locale(val).language3
        md.languageBib = gws.lib.intl.locale(val).languageBib
    else:
        md.language = 'en'
        md.language3 = 'eng'
        md.languageBib = 'eng'


def _update_inspire_theme(md: gws.Metadata, key, val):
    if val:
        md.inspireTheme = val
        md.inspireThemeNameLocal = inspire.theme_name(val, md.language) or ''
        md.inspireThemeNameEn = inspire.theme_name(val, 'en') or ''
    else:
        md.inspireTheme = ''
        md.inspireThemeNameLocal = ''
        md.inspireThemeNameEn = ''


class _Updaters:
    keywords = _update_set
    isoTopicCategories = _update_set
    language = _update_language
    metaLinks = _update_list

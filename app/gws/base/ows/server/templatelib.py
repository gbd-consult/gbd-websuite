"""Helper functions for OWS service templates."""

import gws
import gws.base.ows.server as server
import gws.lib.mime
import gws.lib.uom
import gws.lib.xmlx as xmlx
import gws.types as t


class TemplateArgs(gws.Data):
    featureCollection: server.core.FeatureCollection
    layerCapsTree: server.LayerCapsTree
    layerCapsList: list[server.LayerCaps]
    project: t.Optional[gws.Project]
    request: server.Request
    service: gws.OwsService
    serviceUrl: str
    url_for: t.Callable
    version: str


# OGC 06-121r9 Table 34
# Ordered sequence of two double values in decimal degrees, with longitude before latitude
def ows_wgs84_bounding_box(lc: server.LayerCaps):
    return (
        'ows:WGS84BoundingBox',
        ('ows:LowerCorner', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        ('ows:UpperCorner', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3]))
    )


# OGC 06-121r3 sec 7.4.4
def ows_service_identification(ta: TemplateArgs):
    md = ta.service.metadata

    return (
        'ows:ServiceIdentification',
        ('ows:Title', md.title),
        ('ows:Abstract', md.abstract),
        ows_keywords(md),
        ('ows:ServiceType', ta.service.protocol),
        ('ows:ServiceTypeVersion', ta.version),
        ('ows:Fees', md.fees) if md.fees else None,
        ('ows:AccessConstraints', md.accessConstraints[0].title) if md.accessConstraints else None,
    )


# OGC 06-121r3 sec 7.4.5
def ows_service_provider(ta: TemplateArgs):
    md = ta.service.metadata

    return (
        'ows:ServiceProvider',
        ('ows:ProviderName', md.contactProviderName),
        ('ows:ProviderSite', {'xlink:href': md.contactProviderSite}),
        (
            'ows:ServiceContact',
            ('ows:IndividualName', md.contactPerson),
            ('ows:PositionName', md.contactPosition),
            (
                'ows:ContactInfo',
                (
                    'ows:Phone',
                    ('ows:Voice', md.contactPhone),
                    ('ows:Facsimile', md.contactFax)
                ),
                (
                    'ows:Address',
                    ('ows:DeliveryPoint', md.contactAddress),
                    ('ows:City', md.contactCity),
                    ('ows:AdministrativeArea', md.contactArea),
                    ('ows:PostalCode', md.contactZip),
                    ('ows:Country', md.contactCountry),
                    ('ows:ElectronicMailAddress', md.contactEmail),
                ),
                ('ows:OnlineResource', {'xlink:href': md.contactUrl}),
            ),
            ('ows:Role', md.contactRole)
        )
    )


# OGC 06-121r3 table 15,16,17
# OGC 06-121r3 11.2:
# A URL prefix is defined as a string including... mandatory question mark

def ows_service_url(ta: TemplateArgs, get=True, post=False):
    if get:
        yield 'ows:DCP/ows:HTTP/ows:Get', {'xlink:type': 'simple', 'xlink:href': ta.serviceUrl + '?'}
    if post:
        yield 'ows:DCP/ows:HTTP/ows:Post', {'xlink:type': 'simple', 'xlink:href': ta.serviceUrl}


def ows_value(value):
    return 'ows:Value', value


def online_resource(url):
    return 'OnlineResource', {'xlink:type': 'simple', 'xlink:href': url}


def dcp_service_url(ta: TemplateArgs):
    # OGC 01-068r3, 6.2.2
    # The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
    # OGC 06-042, 6.3.3
    # A URL prefix is defined... as a string including... mandatory question mark
    return 'DCPType/HTTP/Get', online_resource(ta.serviceUrl + '?')


def legend_url(ta: TemplateArgs, lc: server.LayerCaps):
    _, _, name = xmlx.namespace.split_name(lc.layerQname)
    return (
        'LegendURL',
        ('Format', 'image/png'),
        online_resource(f'{ta.serviceUrl}?request=GetLegendGraphic&layer={name}'))


def ows_keywords(md: gws.Metadata):
    return _keywords(md, 'ows:Keywords', 'ows:Keyword')


def keywords(md: gws.Metadata):
    return _keywords(md, 'KeywordList', 'Keyword')


def _keywords(md: gws.Metadata, container_name, tag_name):
    kws = []

    if md.keywords:
        for kw in md.keywords:
            kws.append((tag_name, kw))
    if md.inspireTheme:
        kws.append((tag_name, md.inspireThemeNameEn, {'vocabulary': 'GEMET - INSPIRE themes'}))
    if md.isoTopicCategories:
        for cat in md.isoTopicCategories:
            kws.append((tag_name, cat, {'vocabulary': 'ISO 19115:2003'}))
    if md.inspireMandatoryKeyword:
        kws.append((tag_name, md.inspireMandatoryKeyword, {'vocabulary': 'ISO'}))

    if kws:
        return container_name, kws


def lon_lat_envelope(lc: server.LayerCaps):
    return (
        'lonLatEnvelope',
        {'srsName': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        ('gml:pos', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        ('gml:pos', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
    )


# http://inspire.ec.europa.eu/schemas/common/1.0/network.xsd
# Scenario 2: Mandatory (where appropriate) metadata elements not mapped to standard capabilities,
# plus mandatory language parameters,
# plus OPTIONAL MetadataUrl pointing to an INSPIRE Compliant ISO metadata document

def inspire_extended_capabilities(ta: TemplateArgs):
    md = ta.service.metadata
    return [
        (
            'inspire_common:ResourceLocator',
            ('inspire_common:URL', ta.serviceUrl),
            ('inspire_common:MediaType', 'application/xml'),
        ),
        ('inspire_common:ResourceType', md.inspireResourceType),
        ('inspire_common:TemporalReference/inspire_common:DateOfPublication', md.dateCreated),

        (
            'inspire_common:Conformity',
            (
                'inspire_common:Specification', {'xsi:type': 'inspire_common:citationInspireInteroperabilityRegulation_eng'},
                ('inspire_common:Title', 'COMMISSION REGULATION (EU) No 1089/2010 of 23 November 2010 implementing Directive 2007/2/EC of the European Parliament and of the Council as regards interoperability of spatial data sets and services'),
                ('inspire_common:DateOfPublication', '2010-12-08'),
                ('inspire_common:URI', 'OJ:L:2010:323:0011:0102:EN:PDF'),
                (
                    'inspire_common:ResourceLocator',
                    ('inspire_common:URL', 'http://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=OJ:L:2010:323:0011:0102:EN:PDF'),
                    ('inspire_common:MediaType', 'application/pdf'),
                ),
            ),
            ('inspire_common:Degree', md.inspireDegreeOfConformity),
        ),
        (
            'inspire_common:MetadataPointOfContact',
            ('inspire_common:OrganisationName', md.contactOrganization),
            ('inspire_common:EmailAddress', md.contactEmail),
        ),

        ('inspire_common:MetadataDate', md.dateCreated),
        ('inspire_common:SpatialDataServiceType', md.inspireSpatialDataServiceType),
        ('inspire_common:MandatoryKeyword/inspire_common:KeywordValue', md.inspireMandatoryKeyword),

        (
            'inspire_common:Keyword',
            (
                'inspire_common:OriginatingControlledVocabulary',
                ('inspire_common:Title', 'INSPIRE themes'),
                ('inspire_common:DateOfPublication', '2008-06-01'),
            ),
            ('inspire_common:KeywordValue', md.inspireThemeNameEn),
        ),

        (
            'inspire_common:SupportedLanguages',
            ('inspire_common:DefaultLanguage/inspire_common:Language', md.language3),
            ('inspire_common:SupportedLanguage/inspire_common:Language', md.language3),
        ),

        ('inspire_common:ResponseLanguage/inspire_common:Language', md.language3)
    ]


def coord_dms(n):
    return round(n, gws.lib.uom.DEFAULT_PRECISION[gws.Uom.deg])


def coord_m(n):
    return round(n, gws.lib.uom.DEFAULT_PRECISION[gws.Uom.m])


def to_xml(args, tag):
    ta: TemplateArgs = args.get('owsArgs')
    if ta.request.isSoap:
        tag = 'soap:Envelope', ('soap:Header', ''), ('soap:Body', tag)

    el = xmlx.tag(*tag)
    xml = el.to_string(
        el,
        with_xml_declaration=True,
        with_namespace_declarations=True,
        with_schema_locations=True
    )

    return gws.ContentResponse(mime=gws.lib.mime.XML, content=xml)

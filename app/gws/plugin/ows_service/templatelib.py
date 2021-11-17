"""Helper functions for OWS service templates."""

import gws
import gws.lib.mime
import gws.lib.xml3 as xml3

# OGC 06-121r9 Table 34
# Ordered sequence of two double values in decimal degrees, with longitude before latitude
def ows_wgs84_bounding_box(lc):
    return (
        'ows:WGS84BoundingBox',
        ('ows:LowerCorner', lc.extent4326[0], ' ', lc.extent4326[1]),
        ('ows:UpperCorner', lc.extent4326[2], ' ', lc.extent4326[3]))


# OGC 06-121r3 sec 7.4.4
def ows_service_identification(ARGS):
    md = ARGS.service_meta

    return (
        'ows:ServiceIdentification',
        ('ows:Title', md.title),
        ('ows:Abstract', md.abstract),
        ows_keywords(md),
        ('ows:ServiceType', ARGS.service.protocol),
        ('ows:ServiceTypeVersion', ARGS.version),
        ('ows:Fees', md.fees) if md.feeds else None,
        ('ows:AccessConstraints', md.accessConstraints) if md.accessConstraints else None,
    )


# OGC 06-121r3 sec 7.4.5
def ows_service_provider(ARGS):
    md = ARGS.service_meta

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

def ows_service_url(ARGS, get=True, post=False):
    if get:
        yield 'ows:DCP ows:HTTP ows:Get', {'xlink:type': 'simple', 'xlink:href': ARGS.service_url + '?'}
    if post:
        yield 'ows:DCP ows:HTTP ows:Post', {'xlink:type': 'simple', 'xlink:href': ARGS.service_url}


def online_resource(url):
    return 'OnlineResource', {
        'xlink:type': 'simple',
        'xlink:href': url
    }


def dcp_service_url(ARGS):
    # OGC 01-068r3, 6.2.2
    # The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
    # OGC 06-042, 6.3.3
    # A URL prefix is defined... as a string including... mandatory question mark
    return 'DCPType HTTP Get', online_resource(ARGS.service_url + '?')


def legend_url(ARGS, layer_caps):
    _, name = xml3.split_name(layer_caps.layer_qname)
    return (
        'LegendURL',
        ('Format', 'image/png'),
        online_resource(f'{ARGS.service_url}?request=GetLegendGraphic&layer={name}'))


def ows_keywords(md):
    return _keywords(md, 'ows:Keywords', 'ows:Keyword')


def keywords(md):
    return _keywords(md, 'KeywordList', 'Keyword')


def _keywords(md, container_name, tag_name):
    kws = []

    if md.keywords:
        for kw in md.keywords:
            kws.append((tag_name, kw))
    if md.inspireTheme:
        kws.append((tag_name, md.inspireThemeNameEn, {'vocabulary': 'GEMET - INSPIRE themes'}))
    if md.isoTopicCategory:
        kws.append((tag_name, md.isoTopicCategory, {'vocabulary': 'ISO 19115:2003'}))
    if md.inspireMandatoryKeyword:
        kws.append((tag_name, md.inspireMandatoryKeyword, {'vocabulary': 'ISO'}))

    if kws:
        return container_name, kws


def lon_lat_envelope(lc):
    return (
        'lonLatEnvelope',
        {'srsName': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        ('gml:pos', lc.extent4326[0], ' ', lc.extent4326[1]),
        ('gml:pos', lc.extent4326[2], ' ', lc.extent4326[3]))


# http://inspire.ec.europa.eu/schemas/common/1.0/network.xsd
# Scenario 2: Mandatory (where appropriate) metadata elements not mapped to standard capabilities,
# plus mandatory language parameters,
# plus OPTIONAL MetadataUrl pointing to an INSPIRE Compliant ISO metadata document

def inspire_extended_capabilities(ARGS):
    md = ARGS.service_meta
    return [
        (
            'inspire_common:ResourceLocator',
            ('inspire_common:URL', ARGS.service_url),
            ('inspire_common:MediaType', 'application/xml'),
        ),
        ('inspire_common:ResourceType', md.inspireResourceType),
        ('inspire_common:TemporalReference inspire_common:DateOfPublication', md.dateCreated),

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
        ('inspire_common:MandatoryKeyword inspire_common:KeywordValue', md.inspireMandatoryKeyword),

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
            ('inspire_common:DefaultLanguage inspire_common:Language', md.language3),
            ('inspire_common:SupportedLanguage inspire_common:Language', md.language3),
        ),

        ('inspire_common:ResponseLanguage inspire_common:Language', md.language3)
    ]


def split_name(qname):
    return xml3.split_name(qname)


def to_xml(ARGS, tag):
    if ARGS.with_soap:
        tag = 'soap:Envelope', ('soap:Header', ''), ('soap:Body', tag)
    el = xml3.tag(*tag)
    xml = xml3.to_string(el, with_xml=True, with_xmlns=True, with_schemas=True)
    return gws.ContentResponse(content=xml, mime=gws.lib.mime.XML)

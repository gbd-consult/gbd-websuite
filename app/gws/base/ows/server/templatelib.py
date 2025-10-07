"""Helper functions for OWS service templates."""

from typing import Optional, cast

import gws
import gws.base.metadata
import gws.lib.gml
import gws.lib.uom
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server

from . import core, request, service


# OGC 06-121r9 Table 34
# Ordered sequence of two double values in decimal degrees, with longitude before latitude
def ows_wgs84_bounding_box(lc: core.LayerCaps):
    return (
        'ows:WGS84BoundingBox',
        ('ows:LowerCorner', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        ('ows:UpperCorner', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
    )


# OGC 06-121r3 sec 7.4.4
def ows_service_identification(ta: server.TemplateArgs):
    md = ta.service.metadata

    return (
        'ows:ServiceIdentification',
        ('ows:Title', md.title),
        ('ows:Abstract', md.abstract),
        ows_keywords(md),
        ('ows:ServiceType', ta.service.protocol),
        ('ows:ServiceTypeVersion', ta.version),
        ('ows:Fees', md.fees) if md.fees else None,
        ('ows:AccessConstraints', md.accessConstraints) if md.accessConstraints else None,
    )


# OGC 06-121r3 sec 7.4.5
def ows_service_provider(ta: server.TemplateArgs):
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
                ('ows:Phone', ('ows:Voice', md.contactPhone), ('ows:Facsimile', md.contactFax)),
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
            ('ows:Role', md.contactRole),
        ),
    )


# OGC 06-121r3 table 15,16,17
# OGC 06-121r3 11.2:
# A URL prefix is defined as a string including... mandatory question mark


def ows_service_url(ta: server.TemplateArgs, get=True, post=False):
    if get:
        yield 'ows:DCP/ows:HTTP/ows:Get', {'xlink:type': 'simple', 'xlink:href': ta.serviceUrl + '?'}
    if post:
        yield 'ows:DCP/ows:HTTP/ows:Post', {'xlink:type': 'simple', 'xlink:href': ta.serviceUrl}


def ows_value(value):
    return 'ows:Value', value


def online_resource(url):
    return 'OnlineResource', {'xlink:type': 'simple', 'xlink:href': url}


# OGC 01-068r3, 6.2.2
# The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
# OGC 06-042, 6.3.3
# A URL prefix is defined... as a string including... mandatory question mark


def dcp_service_url(ta: server.TemplateArgs):
    return 'DCPType/HTTP/Get', online_resource(ta.serviceUrl + '?')


def legend_url_nested(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    name = xmlx.namespace.unqualify_name(lc.layerNameQ)
    return (
        'LegendURL',
        {'width': size[0], 'height': size[1]} if size else {},
        ('Format', 'image/png'),
        online_resource(f'{ta.serviceUrl}?request=GetLegendGraphic&layer={name}'),
    )


def legend_url(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    name = xmlx.namespace.unqualify_name(lc.layerNameQ)
    return (
        'LegendURL',
        {
            'format': 'image/png',
            'xlink:href': f'{ta.serviceUrl}?request=GetLegendGraphic&layer={name}',
        },
    )


def ows_keywords(md: gws.Metadata):
    return [_ows_keyword_group(kg) for kg in gws.base.metadata.keyword_groups(md)]


def _ows_keyword_group(kg: gws.base.metadata.KeywordGroup):
    tags = []
    for kw in kg.keywords:
        tags.append(('ows:Keyword', kw))
    if kg.codeSpace:
        tags.append(('ows:Type', {'codeSpace': kg.codeSpace}, kg.typeName))
    return 'ows:Keywords', tags


def wms_keywords(md: gws.Metadata, with_vocabulary: bool = False):
    tags = []
    for kg in gws.base.metadata.keyword_groups(md):
        for kw in kg.keywords:
            if kg.codeSpace and with_vocabulary:
                tags.append(('Keyword', kw, {'vocabulary': kg.codeSpace}))
            else:
                tags.append(('Keyword', kw))
    return 'KeywordList', tags


def lon_lat_envelope(lc: core.LayerCaps):
    return (
        'lonLatEnvelope',
        {'srsName': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        ('gml:pos', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        ('gml:pos', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
    )


# Nested format (WFS 1, WMS):
# OGC 06-042  7.2.4.6.11
# The "type" attribute indicates the standard... The enclosed <Format> element... etc


def meta_links_nested(ta: server.TemplateArgs, md: gws.Metadata):
    if md.metaLinks:
        for ml in md.metaLinks:
            yield meta_url_nested(ta, ml, 'MetadataURL')


def meta_url_nested(ta: server.TemplateArgs, ml: gws.MetadataLink, name: str):
    if ml:
        yield name, {'type': ml.type}, ('Format', ml.format), online_resource(ta.url_for(ml.url))


# Simple format (WFS 2)
# OGC 09-025r1 Table 11
# The xlink:href element shall be used to reference any metadata.
# The optional about attribute may be used to reference the aspect of the element which includes
# this wfs:MetadataURL element that this metadata provides more information about.
# (whatever that means)


def meta_links_simple(ta: server.TemplateArgs, md: gws.Metadata):
    if md.metaLinks:
        for ml in md.metaLinks:
            yield meta_url_simple(ta, ml, 'MetadataURL')


def meta_url_simple(ta: server.TemplateArgs, ml: gws.MetadataLink, name: str):
    if ml:
        yield name, {'xlink:href': ta.url_for(ml.url), 'about': ml.about}


def wfs_feature_collection(ta: server.TemplateArgs):
    return (
        'wfs:FeatureCollection',
        wfs_feature_collection_attributes(ta),
        [
            (
                f'wfs:member/{m.layerCaps.featureNameQ if m.layerCaps else "wfs:feature"}',
                {'gml:id': gml_format_uid(ta, m.feature.uid())},
                wfs_feature_collection_member(ta, m),
            )
            for m in ta.featureCollection.members
        ],
    )


def wfs_value_collection(ta: server.TemplateArgs):
    return (
        'wfs:ValueCollection',
        wfs_feature_collection_attributes(ta),
        [('wfs:member', gml_format_value(ta, val)) for val in ta.featureCollection.values],
    )


def wfs_feature_collection_attributes(ta):
    return {
        'timeStamp': ta.featureCollection.timestamp,
        'numberMatched': ta.featureCollection.numMatched,
        'numberReturned': ta.featureCollection.numReturned,
    }


def wfs_feature_collection_member(ta: server.TemplateArgs, m: server.FeatureCollectionMember):
    geom = None
    for name, val in m.feature.attributes.items():
        if m.layerCaps:
            name = xmlx.namespace.qualify_name(name, m.layerCaps.xmlNamespace)
        if isinstance(val, gws.Shape):
            geom = name, gml_format_value(ta, val)
        else:
            yield name, gml_format_value(ta, val)
    # QGIS wants geometry as the last element
    if geom:
        yield geom


def gml_format_uid(ta: server.TemplateArgs, uid):
    if not uid:
        return '_'
    s = str(uid)
    if s[0].isdigit():
        return '_' + s
    return s


def gml_format_value(ta, val):
    s, ok = xmlx.util.atom_to_string(val)
    if ok:
        return s
    if isinstance(val, gws.Shape):
        # NB Qgis wants inline gml xmlns for adhoc schemas
        return gws.lib.gml.shape_to_element(
            val,
            version=ta.gmlVersion,
            always_xy=ta.sr.alwaysXY,
            with_inline_xmlns=True,
        )
    return str(val)


# http://inspire.ec.europa.eu/schemas/common/1.0/network.xsd
# Scenario 2: Mandatory (where appropriate) metadata elements not mapped to standard capabilities,
# plus mandatory language parameters,
# plus OPTIONAL MetadataUrl pointing to an INSPIRE Compliant ISO metadata document


def inspire_extended_capabilities(ta: server.TemplateArgs):
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
                'inspire_common:Specification',
                # {'xsi:type': 'inspire_common:citationInspireInteroperabilityRegulation'},
                (
                    'inspire_common:Title',
                    'COMMISSION REGULATION (EU) No 1089/2010 of 23 November 2010 implementing Directive 2007/2/EC of the European Parliament and of the Council as regards interoperability of spatial data sets and services',
                ),
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
            ('inspire_common:DefaultLanguage/inspire_common:Language', md.languageBib),
            ('inspire_common:SupportedLanguage/inspire_common:Language', md.languageBib),
        ),
        ('inspire_common:ResponseLanguage/inspire_common:Language', md.languageBib),
    ]


def coord_dms(n):
    return round(n, gws.lib.uom.DEFAULT_PRECISION[gws.Uom.deg])


def coord_m(n):
    return round(n, gws.lib.uom.DEFAULT_PRECISION[gws.Uom.m])


def namespaces_from_caps(ta: server.TemplateArgs) -> dict[str, gws.XmlNamespace]:
    return {lc.xmlNamespace.xmlns: lc.xmlNamespace for lc in ta.layerCapsList if lc.xmlNamespace is not None}


def to_xml_response(
    ta: server.TemplateArgs,
    tag,
    namespaces: Optional[dict[str, gws.XmlNamespace]] = None,
    default_namespace: Optional[gws.XmlNamespace] = None,
) -> gws.ContentResponse:
    if ta.sr.isSoap:
        tag = ['soap:Envelope', ('soap:Header', ''), ('soap:Body', tag)]
        namespaces = namespaces or {}
        namespaces['soap'] = xmlx.namespace.require('soap')

    el = xmlx.tag(*tag)
    opts = gws.XmlOptions(
        namespaces=namespaces,
        defaultNamespace=default_namespace,
        withNamespaceDeclarations=True,
        withSchemaLocations=True,
        withXmlDeclaration=True,
        xmlnsReplacements=ta.sr.xmlnsReplacements,
    )

    return cast(service.Object, ta.sr.service).xml_response(el, opts)


def to_xml_response_with_doctype(
    ta: server.TemplateArgs,
    tag,
    doctype: str,
) -> gws.ContentResponse:
    if ta.sr.isSoap:
        tag = ['soap:Envelope', ('soap:Header', ''), ('soap:Body', tag)]

    el = xmlx.tag(*tag)
    opts = gws.XmlOptions(
        doctype=doctype,
        withNamespaceDeclarations=False,
        withSchemaLocations=False,
        withXmlDeclaration=True,
    )
    return cast(service.Object, ta.sr.service).xml_response(el, opts)

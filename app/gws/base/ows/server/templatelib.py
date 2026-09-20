"""Helper functions for OWS service templates."""

from typing import Optional, cast

import gws
import gws.base.metadata
import gws.lib.gml
import gws.lib.uom
import gws.lib.datetimex as dtx
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server

from . import core, service
from gws.lib.xmlx import tag

OWS = 'OWS_11'
"""Default OWS namespace uid for the ``ows_*`` helpers."""


# OGC 06-121r9 Table 34
# Ordered sequence of two double values in decimal degrees, with longitude before latitude
def ows_wgs84_bounding_box(lc: core.LayerCaps, ows: str = OWS):
    return tag(
        f'{ows}:WGS84BoundingBox',
        tag(f'{ows}:LowerCorner', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        tag(f'{ows}:UpperCorner', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
    )


# OGC 06-121r3 sec 7.4.4
def ows_service_identification(ta: server.TemplateArgs, ows: str = OWS):
    md = ta.service.metadata

    return tag(
        f'{ows}:ServiceIdentification',
        tag(f'{ows}:Title', md.title),
        tag(f'{ows}:Abstract', md.abstract),
        ows_keywords(md, ows),
        tag(f'{ows}:ServiceType', ta.service.protocol),
        tag(f'{ows}:ServiceTypeVersion', ta.version),
        tag(f'{ows}:Fees', md.fees) if md.fees else None,
        tag(f'{ows}:AccessConstraints', md.accessConstraints) if md.accessConstraints else None,
    )


# OGC 06-121r3 sec 7.4.5
def ows_service_provider(ta: server.TemplateArgs, ows: str = OWS):
    md = ta.service.metadata

    return tag(
        f'{ows}:ServiceProvider',
        tag(f'{ows}:ProviderName', md.contactProviderName),
        tag(f'{ows}:ProviderSite', {'XLINK:href': md.contactProviderSite}),
        tag(
            f'{ows}:ServiceContact',
            tag(f'{ows}:IndividualName', md.contactPerson),
            tag(f'{ows}:PositionName', md.contactPosition),
            tag(
                f'{ows}:ContactInfo',
                tag(f'{ows}:Phone', tag(f'{ows}:Voice', md.contactPhone), tag(f'{ows}:Facsimile', md.contactFax)),
                tag(
                    f'{ows}:Address',
                    tag(f'{ows}:DeliveryPoint', md.contactAddress),
                    tag(f'{ows}:City', md.contactCity),
                    tag(f'{ows}:AdministrativeArea', md.contactArea),
                    tag(f'{ows}:PostalCode', md.contactZip),
                    tag(f'{ows}:Country', md.contactCountry),
                    tag(f'{ows}:ElectronicMailAddress', md.contactEmail),
                ),
                tag(f'{ows}:OnlineResource', {'XLINK:href': md.contactUrl}),
            ),
            tag(f'{ows}:Role', md.contactRole),
        ),
    )


# OGC 06-121r3 table 15,16,17
# OGC 06-121r3 11.2:
# A URL prefix is defined as a string including... mandatory question mark


def ows_service_url(ta: server.TemplateArgs, get=True, post=False, ows: str = OWS):
    if get:
        yield tag(f'{ows}:DCP/{ows}:HTTP/{ows}:Get', {'XLINK:type': 'simple', 'XLINK:href': ta.serviceUrl + '?'})
    if post:
        yield tag(f'{ows}:DCP/{ows}:HTTP/{ows}:Post', {'XLINK:type': 'simple', 'XLINK:href': ta.serviceUrl})


def ows_value(value, ows: str = OWS):
    return tag(f'{ows}:Value', value)


def online_resource(url):
    return tag('OnlineResource', {'XLINK:type': 'simple', 'XLINK:href': url})


# OGC 01-068r3, 6.2.2
# The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
# OGC 06-042, 6.3.3
# A URL prefix is defined... as a string including... mandatory question mark


def dcp_service_url(ta: server.TemplateArgs):
    return tag('DCPType/HTTP/Get', online_resource(ta.serviceUrl + '?'))


def legend_url_nested(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    name = xmlx.namespace.unqualify_name(lc.layerNameQ)
    return tag(
        'LegendURL',
        {'width': size[0], 'height': size[1]} if size else {},
        tag('Format', 'image/png'),
        online_resource(f'{ta.serviceUrl}?request=GetLegendGraphic&layer={name}'),
    )


def legend_url(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    name = xmlx.namespace.unqualify_name(lc.layerNameQ)
    return tag(
        'LegendURL',
        {
            'format': 'image/png',
            'XLINK:href': f'{ta.serviceUrl}?request=GetLegendGraphic&layer={name}',
        },
    )


def ows_keywords(md: gws.Metadata, ows: str = OWS):
    return [_ows_keyword_group(kg, ows) for kg in gws.base.metadata.keyword_groups(md)]


def _ows_keyword_group(kg: gws.base.metadata.KeywordGroup, ows: str):
    tags = []
    for kw in kg.keywords:
        tags.append(tag(f'{ows}:Keyword', kw))
    if kg.codeSpace:
        tags.append(tag(f'{ows}:Type', {'codeSpace': kg.codeSpace}, kg.typeName))
    return tag(f'{ows}:Keywords', tags)


def wms_keywords(md: gws.Metadata, with_vocabulary: bool = False):
    tags = []
    for kg in gws.base.metadata.keyword_groups(md):
        for kw in kg.keywords:
            if kg.codeSpace and with_vocabulary:
                tags.append(tag('Keyword', kw, {'vocabulary': kg.codeSpace}))
            else:
                tags.append(tag('Keyword', kw))
    return tag('KeywordList', tags)


def lon_lat_envelope(lc: core.LayerCaps):
    return tag(
        'lonLatEnvelope',
        {'srsName': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
        tag('GML:pos', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        tag('GML:pos', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
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
        yield tag(name, {'type': ml.type}, tag('Format', ml.format), online_resource(ta.url_for(ml.url)))


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
        yield tag(name, {'XLINK:href': ta.url_for(ml.url), 'about': ml.about})


def wfs_feature_collection(ta: server.TemplateArgs):
    return tag(
        'WFS:FeatureCollection',
        wfs_feature_collection_attributes(ta),
        [
            tag(
                'WFS:member',
                tag(
                    xmlx.namespace.clark_name(m.layerCaps.featureName, m.layerCaps.xmlNamespace) if m.layerCaps else 'WFS:feature',
                    {'GML:id': gml_format_uid(ta, m.feature.uid())},
                    wfs_feature_collection_member(ta, m),
                ),
            )
            for m in ta.featureCollection.members
        ],
    )


def wfs_value_collection(ta: server.TemplateArgs):
    return tag(
        'WFS:ValueCollection',
        wfs_feature_collection_attributes(ta),
        [tag('WFS:member', gml_format_value(ta, val)) for val in ta.featureCollection.values],
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
            name = xmlx.namespace.clark_name(name, m.layerCaps.xmlNamespace)
        if isinstance(val, gws.Shape):
            geom = tag(name, gml_format_value(ta, val))
        else:
            yield tag(name, gml_format_value(ta, val))
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
        tag(
            'INSPIRE_COMMON:ResourceLocator',
            tag('INSPIRE_COMMON:URL', ta.serviceUrl),
            tag('INSPIRE_COMMON:MediaType', 'application/xml'),
        ),
        tag('INSPIRE_COMMON:ResourceType', md.inspireResourceType),
        tag('INSPIRE_COMMON:TemporalReference/INSPIRE_COMMON:DateOfPublication', iso_date(md.dateCreated)),
        tag(
            'INSPIRE_COMMON:Conformity',
            tag(
                'INSPIRE_COMMON:Specification',
                # {'XSI:type': 'inspire_common:citationInspireInteroperabilityRegulation'},
                tag(
                    'INSPIRE_COMMON:Title',
                    'COMMISSION REGULATION (EU) No 1089/2010 of 23 November 2010 implementing Directive 2007/2/EC of the European Parliament and of the Council as regards interoperability of spatial data sets and services',
                ),
                tag('INSPIRE_COMMON:DateOfPublication', '2010-12-08'),
                tag('INSPIRE_COMMON:URI', 'OJ:L:2010:323:0011:0102:EN:PDF'),
                tag(
                    'INSPIRE_COMMON:ResourceLocator',
                    tag('INSPIRE_COMMON:URL', 'http://eur-lex.europa.eu/LexUriServ/LexUriServ.do?uri=OJ:L:2010:323:0011:0102:EN:PDF'),
                    tag('INSPIRE_COMMON:MediaType', 'application/pdf'),
                ),
            ),
            tag('INSPIRE_COMMON:Degree', md.inspireDegreeOfConformity),
        ),
        tag(
            'INSPIRE_COMMON:MetadataPointOfContact',
            tag('INSPIRE_COMMON:OrganisationName', md.contactOrganization),
            tag('INSPIRE_COMMON:EmailAddress', md.contactEmail),
        ),
        tag('INSPIRE_COMMON:MetadataDate', iso_date(md.dateCreated)),
        tag('INSPIRE_COMMON:SpatialDataServiceType', md.inspireSpatialDataServiceType),
        tag('INSPIRE_COMMON:MandatoryKeyword/INSPIRE_COMMON:KeywordValue', md.inspireMandatoryKeyword),
        tag(
            'INSPIRE_COMMON:Keyword',
            tag(
                'INSPIRE_COMMON:OriginatingControlledVocabulary',
                tag('INSPIRE_COMMON:Title', 'INSPIRE themes'),
                tag('INSPIRE_COMMON:DateOfPublication', '2008-06-01'),
            ),
            tag('INSPIRE_COMMON:KeywordValue', md.inspireThemeNameEn),
        ),
        tag(
            'INSPIRE_COMMON:SupportedLanguages',
            tag('INSPIRE_COMMON:DefaultLanguage/INSPIRE_COMMON:Language', md.languageBib),
            tag('INSPIRE_COMMON:SupportedLanguage/INSPIRE_COMMON:Language', md.languageBib),
        ),
        tag('INSPIRE_COMMON:ResponseLanguage/INSPIRE_COMMON:Language', md.languageBib),
    ]


def coord_dms(n):
    prec = gws.lib.uom.DEFAULT_PRECISION[gws.Uom.deg]
    return f'{round(n, prec):.{prec}f}'


def coord_m(n):
    prec = gws.lib.uom.DEFAULT_PRECISION[gws.Uom.m]
    return f'{round(n, prec):.{prec}f}'


def iso_date(d):
    dd = dtx.parse(d)
    return dtx.to_iso_date_string(dd) if dd else ''


def iso_datetime(d):
    dd = dtx.parse(d)
    return dtx.to_iso_string(dd, with_tz=':') if dd else ''


def namespaces_from_caps(ta: server.TemplateArgs) -> list[gws.XmlNamespace]:
    """Feature type namespaces, to declare in documents that reference feature types by QName."""

    d = {}
    for lc in ta.layerCapsList:
        if lc.xmlNamespace:
            d[lc.xmlNamespace.uri] = lc.xmlNamespace
    return list(d.values())


def to_xml_response(
    ta: server.TemplateArgs,
    el: gws.XmlElement,
    default_namespace: Optional[gws.XmlNamespace] = None,
    namespaces: Optional[list[gws.XmlNamespace]] = None,
) -> gws.ContentResponse:
    """Create an XML response.

    Args:
        ta: Template arguments.
        el: Root element.
        default_namespace: Default namespace of the document.
        namespaces: Namespaces to declare on the root in addition to those used by elements and attributes,
            e.g. for QName values.
    """

    if namespaces:
        el.namespaces.extend(namespaces)

    if ta.sr.isSoap:
        el = tag('SOAP:Envelope', tag('SOAP:Header'), tag('SOAP:Body', el))

    opts = gws.XmlOptions(
        defaultNamespace=default_namespace,
        withNamespaceDeclarations=True,
        withSchemaLocations=True,
        withXmlDeclaration=True,
        customXmlns=ta.sr.customXmlns,
    )

    return cast(service.Object, ta.sr.service).xml_response(el, opts)


def to_xml_response_with_doctype(
    ta: server.TemplateArgs,
    el: gws.XmlElement,
    doctype: str,
) -> gws.ContentResponse:
    if ta.sr.isSoap:
        el = tag('SOAP:Envelope', tag('SOAP:Header'), tag('SOAP:Body', el))

    # DTD-based formats have no namespace declarations on the root,
    # declare xlink inline where it is used (as per the DTD's #FIXED xmlns:xlink)
    xlink = xmlx.namespace.ns.XLINK
    xlink_prefix = '{' + xlink.uri + '}'
    for e in el.iter():
        if any(k.startswith(xlink_prefix) for k in e.attrib):
            e.namespaces.append(xlink)

    opts = gws.XmlOptions(
        doctype=doctype,
        withNamespaceDeclarations=False,
        withSchemaLocations=False,
        withXmlDeclaration=True,
    )
    return cast(service.Object, ta.sr.service).xml_response(el, opts)

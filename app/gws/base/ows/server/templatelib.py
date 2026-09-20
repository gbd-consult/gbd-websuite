"""Helper functions for OWS service templates."""

from typing import Optional, cast

import gws
import gws.base.metadata
import gws.lib.gml
import gws.lib.uom
import gws.lib.datetimex as dtx
import gws.lib.xmlx as xmlx
import gws.base.ows.server as server

from . import core, layer_caps, service
from gws.lib.xmlx import tag

# OGC 06-121r9 Table 34
# Ordered sequence of two double values in decimal degrees, with longitude before latitude
def ows_wgs84_bounding_box(lc: core.LayerCaps, ns: str = 'OWS_11'):
    return tag(
        f'{ns}:WGS84BoundingBox',
        tag(f'{ns}:LowerCorner', coord_dms(lc.layer.wgsExtent[0]), ' ', coord_dms(lc.layer.wgsExtent[1])),
        tag(f'{ns}:UpperCorner', coord_dms(lc.layer.wgsExtent[2]), ' ', coord_dms(lc.layer.wgsExtent[3])),
    )


# OGC 06-121r3 sec 7.4.4
def ows_service_identification(ta: server.TemplateArgs, ns: str = 'OWS_11'):
    md = ta.service.metadata

    return tag(
        f'{ns}:ServiceIdentification',
        tag(f'{ns}:Title', md.title),
        tag(f'{ns}:Abstract', md.abstract),
        ows_keywords(md, ns),
        tag(f'{ns}:ServiceType', ta.service.protocol),
        tag(f'{ns}:ServiceTypeVersion', ta.version),
        tag(f'{ns}:Fees', md.fees) if md.fees else None,
        tag(f'{ns}:AccessConstraints', md.accessConstraints) if md.accessConstraints else None,
    )


# OGC 06-121r3 sec 7.4.5
def ows_service_provider(ta: server.TemplateArgs, ns: str = 'OWS_11'):
    md = ta.service.metadata

    return tag(
        f'{ns}:ServiceProvider',
        tag(f'{ns}:ProviderName', md.contactProviderName),
        tag(f'{ns}:ProviderSite', {'XLINK:href': md.contactProviderSite}),
        tag(
            f'{ns}:ServiceContact',
            tag(f'{ns}:IndividualName', md.contactPerson),
            tag(f'{ns}:PositionName', md.contactPosition),
            tag(
                f'{ns}:ContactInfo',
                tag(
                    f'{ns}:Phone',
                    tag(f'{ns}:Voice', md.contactPhone),
                    tag(f'{ns}:Facsimile', md.contactFax),
                ),
                tag(
                    f'{ns}:Address',
                    tag(f'{ns}:DeliveryPoint', md.contactAddress),
                    tag(f'{ns}:City', md.contactCity),
                    tag(f'{ns}:AdministrativeArea', md.contactArea),
                    tag(f'{ns}:PostalCode', md.contactZip),
                    tag(f'{ns}:Country', md.contactCountry),
                    tag(f'{ns}:ElectronicMailAddress', md.contactEmail),
                ),
                tag(f'{ns}:OnlineResource', {'XLINK:href': md.contactUrl}),
            ),
            tag(f'{ns}:Role', md.contactRole),
        ),
    )


# OGC 06-121r3 table 15,16,17
# OGC 06-121r3 11.2:
# A URL prefix is defined as a string including... mandatory question mark


def ows_service_url(ta: server.TemplateArgs, get=True, post=False, ns: str = 'OWS_11'):
    if get:
        yield tag(
            f'{ns}:DCP/{ns}:HTTP/{ns}:Get',
            {'XLINK:type': 'simple', 'XLINK:href': ta.serviceUrl + '?'},
        )
    if post:
        yield tag(
            f'{ns}:DCP/{ns}:HTTP/{ns}:Post',
            {'XLINK:type': 'simple', 'XLINK:href': ta.serviceUrl},
        )


def ows_value(value, ns: str = 'OWS_11'):
    return tag(f'{ns}:Value', value)


def online_resource(url):
    return tag('OnlineResource', {'XLINK:type': 'simple', 'XLINK:href': url})


# OGC 01-068r3, 6.2.2
# The URL prefix shall end in either a '?' (in the absence of additional server-specific parameters) or a '&'.
# OGC 06-042, 6.3.3
# A URL prefix is defined... as a string including... mandatory question mark


def dcp_service_url(ta: server.TemplateArgs):
    return tag('DCPType/HTTP/Get', online_resource(ta.serviceUrl + '?'))


def legend_url_nested(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    return tag(
        'LegendURL',
        {'width': size[0], 'height': size[1]} if size else {},
        tag('Format', 'image/png'),
        online_resource(f'{ta.serviceUrl}?request=GetLegendGraphic&layer={lc.layerName}'),
    )


def legend_url(ta: server.TemplateArgs, lc: core.LayerCaps, size=None):
    return tag(
        'LegendURL',
        {
            'format': 'image/png',
            'XLINK:href': f'{ta.serviceUrl}?request=GetLegendGraphic&layer={lc.layerName}',
        },
    )


def feature_name(ta: server.TemplateArgs, lc: core.LayerCaps) -> str:
    return layer_caps.qualified_feature_name(lc, ta.serviceRequest.customNamespacePrefixes)


def ows_keywords(md: gws.Metadata, ns: str = 'OWS_11'):
    return [_ows_keyword_group(kg, ns) for kg in gws.base.metadata.keyword_groups(md)]


def _ows_keyword_group(kg: gws.base.metadata.KeywordGroup, ns: str):
    tags = []
    for kw in kg.keywords:
        tags.append(tag(f'{ns}:Keyword', kw))
    if kg.codeSpace:
        tags.append(tag(f'{ns}:Type', {'codeSpace': kg.codeSpace}, kg.typeName))
    return tag(f'{ns}:Keywords', tags)


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
                    xmlx.namespace.full_name(m.layerCaps.featureName, m.layerCaps.xmlNamespace) if m.layerCaps else 'WFS:feature',
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
            name = xmlx.namespace.full_name(name, m.layerCaps.xmlNamespace)
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
            always_xy=ta.serviceRequest.alwaysXY,
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
    extra_namespaces: Optional[list[gws.XmlNamespace]] = None,
    doctype: Optional[str] = None,
) -> gws.ContentResponse:
    """Create an XML response.

    Args:
        ta: Template arguments.
        el: Root element.
        default_namespace: Default namespace of the document.
        extra_namespaces: Namespaces to declare on the root in addition to those used by elements and attributes,
            e.g. for QName values.
        doctype: DTD for DTD-based formats (WMS 1.1.x). These have no namespace declarations on the root;
            ``xlink`` is declared inline where it is used (as per the DTD's #FIXED xmlns:xlink).
    """

    if extra_namespaces:
        el.declare(*extra_namespaces)

    if doctype:
        xlink = xmlx.namespace.c.XLINK
        xlink_uri = '{' + xlink.uri + '}'
        for e in el.iter():
            if any(k.startswith(xlink_uri) for k in e.attrib):
                e.declare(xlink)

    if ta.serviceRequest.isSoap:
        el = tag('SOAP:Envelope', tag('SOAP:Header'), tag('SOAP:Body', el))

    opts = gws.XmlOptions(
        doctype=doctype,
        defaultNamespace=default_namespace,
        withNamespaceDeclarations=not doctype,
        withSchemaLocations=not doctype,
        withXmlDeclaration=True,
        customNamespacePrefixes=ta.serviceRequest.customNamespacePrefixes,
    )

    return cast(service.Object, ta.serviceRequest.service).xml_response(el, opts)

from typing import cast
import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.uom
import gws.lib.extent
import gws.lib.xmlx as xmlx
import gws.plugin.ows_server.wms
from gws.lib.xmlx import tag

_DOCTYPE_110 = 'WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.0/capabilities_1_1_0.dtd"'
_DOCTYPE_111 = 'WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/capabilities_1_1_1.dtd"'


def main(ta: server.TemplateArgs):
    if ta.intVersion == 130:
        return tpl.to_xml_response(
            ta,
            tag('WMS_Capabilities', doc(ta)),
            default_namespace=xmlx.namespace.require('wms'),
        )

    if ta.intVersion == 110:
        return tpl.to_xml_response_with_doctype(
            ta,
            tag('WMT_MS_Capabilities', doc(ta)),
            doctype=_DOCTYPE_110,
        )

    if ta.intVersion == 111:
        return tpl.to_xml_response_with_doctype(
            ta,
            tag('WMT_MS_Capabilities', doc(ta)),
            doctype=_DOCTYPE_111,
        )

    return tpl.to_xml_response(ta, tag('WMS_Capabilities', doc(ta)))


def doc(ta):
    yield {
        'version': ta.version,
        'updateSequence': ta.service.updateSequence,
    }

    yield tag('Service', service_meta(ta))
    yield tag('Capability', caps(ta))


def service_meta(ta):
    md = ta.service.metadata

    yield tag('Name', 'WMS')
    yield tag('Title', md.title)
    yield tag('Abstract', md.abstract)

    yield tpl.wms_keywords(md)
    yield tpl.online_resource(ta.serviceUrl)

    yield tag(
        'ContactInformation',
        tag('ContactPersonPrimary', tag('ContactPerson', md.contactPerson), tag('ContactOrganization', md.contactOrganization)),
        tag('ContactPosition', md.contactPosition),
        tag(
            'ContactAddress',
            tag('AddressType', 'postal'),
            tag('Address', md.contactAddress),
            tag('City', md.contactCity),
            tag('StateOrProvince', md.contactArea),
            tag('PostCode', md.contactZip),
            tag('Country', md.contactCountry),
        ),
        tag('ContactVoiceTelephone', md.contactPhone),
        tag('ContactElectronicMailAddress', md.contactEmail),
    )

    if md.fees:
        yield tag('Fees', md.fees)

    if md.accessConstraints:
        yield tag('AccessConstraints', md.accessConstraints)

    if ta.intVersion == 130:
        s = cast(gws.plugin.ows_server.wms.Object, ta.service).layerLimit
        if s:
            yield tag('LayerLimit', s)

        s = cast(gws.plugin.ows_server.wms.Object, ta.service).maxPixelSize
        if s:
            yield tag('MaxWidth', s)
            yield tag('MaxHeight', s)

    yield tpl.meta_links_nested(ta, md)


def caps(ta):
    yield tag('Request', request_caps(ta))

    yield tag('Exception/Format', 'XML')

    if ta.service.withInspireMeta:
        if ta.intVersion == 130:
            yield tag('inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta))
        else:
            yield tag('VendorSpecificCapabilities/inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta))

    yield layer(ta, ta.layerCapsList[0])


def request_caps(ta):
    url = tpl.dcp_service_url(ta)

    for op in ta.service.supportedOperations:
        verb = op.verb
        if verb == gws.OwsVerb.GetLegendGraphic:
            if ta.intVersion == 130:
                verb = 'sld:GetLegendGraphic'
            if ta.intVersion == 110:
                continue
        # NB QGIS wants a space after ';'
        yield tag(verb, [tag('Format', f.replace(';', '; ')) for f in op.formats], url)


def layer(ta, lc: server.LayerCaps):
    return tag('Layer', {'queryable': 1 if lc.isSearchable else 0}, layer_content(ta, lc))


def layer_content(ta, lc: server.LayerCaps):
    md = lc.layer.metadata

    yield tag('Name', lc.layerName)
    yield tag('Title', lc.title)
    yield tag('Abstract', md.abstract)

    yield tpl.wms_keywords(md, with_vocabulary=ta.intVersion == 130)

    wext = lc.layer.wgsExtent
    crs = 'CRS' if ta.intVersion == 130 else 'SRS'

    for b in lc.bounds:
        yield tag(crs, b.crs.epsg)
        if ta.intVersion == 110:
            break

    if ta.intVersion == 130:
        yield tag(
            'EX_GeographicBoundingBox',
            tag('westBoundLongitude', tpl.coord_dms(wext[0])),
            tag('eastBoundLongitude', tpl.coord_dms(wext[2])),
            tag('southBoundLatitude', tpl.coord_dms(wext[1])),
            tag('northBoundLatitude', tpl.coord_dms(wext[3])),
        )
    else:
        # OGC 01-068r3, 6.5.6
        # When the SRS is a Platte Carrée projection of longitude and latitude coordinates,
        # X refers to the longitudinal axis and Y to the latitudinal axis.
        yield tag(
            'LatLonBoundingBox',
            {
                'minx': tpl.coord_dms(wext[0]),
                'miny': tpl.coord_dms(wext[1]),
                'maxx': tpl.coord_dms(wext[2]),
                'maxy': tpl.coord_dms(wext[3]),
            },
        )

    for b in lc.bounds:
        bext = b.extent
        if b.crs.isYX and ta.intVersion == 130:
            bext = gws.lib.extent.swap_xy(bext)
        fn = tpl.coord_dms if b.crs.isGeographic else tpl.coord_m
        yield tag(
            'BoundingBox',
            {
                crs: b.crs.epsg,
                'minx': fn(bext[0]),
                'miny': fn(bext[1]),
                'maxx': fn(bext[2]),
                'maxy': fn(bext[3]),
            },
        )

    if md.attribution:
        yield tag(
            'Attribution',
            tag('Title', md.attribution),
            tpl.online_resource(md.attributionUrl) if md.attributionUrl else None,
        )

    if md.authorityUrl:
        yield tag('AuthorityURL', {'name': md.authorityName}, tpl.online_resource(md.authorityUrl))

    if md.authorityIdentifier:
        yield tag('Identifier', {'authority': md.authorityName}, md.authorityIdentifier)

    yield tpl.meta_links_nested(ta, md)

    if lc.hasLegend:
        if ta.intVersion == 130:
            u = tpl.legend_url_nested(ta, lc)
        else:
            # @TODO compute the size somehow?
            u = tpl.legend_url_nested(ta, lc, size=[256, 256])

        yield tag(
            'Style',
            tag('Name', 'default'),
            tag('Title', 'default'),
            u,
        )

    if not lc.children:
        if ta.intVersion == 130:
            yield tag('MinScaleDenominator', lc.minScale)
            yield tag('MaxScaleDenominator', lc.maxScale)
        else:
            # OGC 01-068r3, 7.1.4.5.8
            # the diagonal with 1 meter per pixel
            diag = (2**0.5) * gws.lib.uom.OGC_M_PER_PX
            yield tag(
                'ScaleHint',
                {
                    'min': lc.minScale * diag,
                    'max': lc.maxScale * diag,
                },
            )

    for c in lc.children:
        yield layer(ta, c)

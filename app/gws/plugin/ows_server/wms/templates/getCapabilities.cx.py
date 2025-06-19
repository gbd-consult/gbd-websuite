from typing import cast
import gws
import gws.base.ows.server as server
import gws.base.ows.server.templatelib as tpl
import gws.lib.uom
import gws.lib.extent
import gws.plugin.ows_server.wms


def main(ta: server.TemplateArgs):
    if ta.intVersion == 130:
        return tpl.to_xml_response(ta, ('WMS_Capabilities', doc(ta)))

    if ta.intVersion == 110:
        return tpl.to_xml_response_with_doctype(
            ta,
            ('WMT_MS_Capabilities', doc(ta)),
            doctype='WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.0/capabilities_1_1_0.dtd"',
        )
    
    if ta.intVersion == 111:
        return tpl.to_xml_response_with_doctype(
            ta,
            ('WMT_MS_Capabilities', doc(ta)),
            doctype='WMT_MS_Capabilities SYSTEM "http://schemas.opengis.net/wms/1.1.1/WMS_MS_Capabilities.dtd"',
        )

    return tpl.to_xml_response(ta, ('WMS_Capabilities', doc(ta)))




def doc(ta):
    yield {
        'version': ta.version,
        'updateSequence': ta.service.updateSequence,
    }
    if ta.intVersion == 130:
        yield {'xmlns': 'wms'}

    yield 'Service', service_meta(ta)
    yield 'Capability', caps(ta)


def service_meta(ta):
    md = ta.service.metadata

    yield 'Name', 'WMS'
    yield 'Title', md.title
    yield 'Abstract', md.abstract

    yield tpl.wms_keywords(md)
    yield tpl.online_resource(ta.serviceUrl)

    yield (
        'ContactInformation',
        ('ContactPersonPrimary', ('ContactPerson', md.contactPerson), ('ContactOrganization', md.contactOrganization)),
        ('ContactPosition', md.contactPosition),
        (
            'ContactAddress',
            ('AddressType', 'postal'),
            ('Address', md.contactAddress),
            ('City', md.contactCity),
            ('StateOrProvince', md.contactArea),
            ('PostCode', md.contactZip),
            ('Country', md.contactCountry),
        ),
        ('ContactVoiceTelephone', md.contactPhone),
        ('ContactElectronicMailAddress', md.contactEmail),
    )

    if md.fees:
        yield 'Fees', md.fees

    if md.accessConstraints:
        yield 'AccessConstraints', md.accessConstraints

    if ta.intVersion == 130:
        s = cast(gws.plugin.ows_server.wms.Object, ta.service).layerLimit
        if s:
            yield 'LayerLimit', s

        s = cast(gws.plugin.ows_server.wms.Object, ta.service).maxPixelSize
        if s:
            yield 'MaxWidth', s
            yield 'MaxHeight', s

    yield tpl.meta_links_nested(ta, md)


def caps(ta):
    yield 'Request', request_caps(ta)

    yield 'Exception/Format', 'XML'

    if ta.service.withInspireMeta:
        if ta.intVersion == 130:
            yield 'inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)
        else:
            yield 'VendorSpecificCapabilities/inspire_vs:ExtendedCapabilities', tpl.inspire_extended_capabilities(ta)

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
        yield verb, [('Format', f.replace(';', '; ')) for f in op.formats], url


def layer(ta, lc: server.LayerCaps):
    return 'Layer', {'queryable': 1 if lc.isSearchable else 0}, layer_content(ta, lc)


def layer_content(ta, lc: server.LayerCaps):
    md = lc.layer.metadata

    yield 'Name', lc.layerName
    yield 'Title', lc.title
    yield 'Abstract', md.abstract

    yield tpl.wms_keywords(md, with_vocabulary=ta.intVersion == 130)

    wext = lc.layer.wgsExtent
    crs = 'CRS' if ta.intVersion == 130 else 'SRS'

    
    for b in lc.bounds:
        yield crs, b.crs.epsg
        if ta.intVersion == 110:
            break

    if ta.intVersion == 130:
        yield (
            'EX_GeographicBoundingBox',
            ('westBoundLongitude', tpl.coord_dms(wext[0])),
            ('eastBoundLongitude', tpl.coord_dms(wext[2])),
            ('southBoundLatitude', tpl.coord_dms(wext[1])),
            ('northBoundLatitude', tpl.coord_dms(wext[3])),
        )
    else:
        # OGC 01-068r3, 6.5.6
        # When the SRS is a Platte Carrée projection of longitude and latitude coordinates,
        # X refers to the longitudinal axis and Y to the latitudinal axis.
        yield (
            'LatLonBoundingBox',
            {
                'minx': wext[0],
                'miny': wext[1],
                'maxx': wext[2],
                'maxy': wext[3],
            },
        )

    for b in lc.bounds:
        bext = b.extent
        if b.crs.isYX and ta.intVersion == 130:
            bext = gws.lib.extent.swap_xy(bext)
        fn = tpl.coord_dms if b.crs.isGeographic else tpl.coord_m
        yield (
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
        yield (
            'Attribution',
            ('Title', md.attribution),
            tpl.online_resource(md.attributionUrl) if md.attributionUrl else None,
        )

    if md.authorityUrl:
        yield 'AuthorityURL', {'name': md.authorityName}, tpl.online_resource(md.authorityUrl)

    if md.authorityIdentifier:
        yield 'Identifier', {'authority': md.authorityName}, md.authorityIdentifier

    yield tpl.meta_links_nested(ta, md)

    if lc.hasLegend:
        yield (
            'Style',
            ('Name', 'default'),
            ('Title', 'default'),
            tpl.legend_url(ta, lc, None if ta.intVersion == 130 else [256, 256]),
        )

    if not lc.children:
        if ta.intVersion == 130:
            yield 'MinScaleDenominator', lc.minScale
            yield 'MaxScaleDenominator', lc.maxScale
        else:
            # OGC 01-068r3, 7.1.4.5.8
            # the diagonal with 1 meter per pixel
            diag = (2**0.5) * gws.lib.uom.OGC_M_PER_PX
            yield (
                'ScaleHint',
                {
                    'min': lc.minScale * diag,
                    'max': lc.maxScale * diag,
                },
            )

    for c in lc.children:
        yield layer(ta, c)

"""WMS Capabilities parser."""

from typing import Optional

import gws
import gws.base.ows.client
import gws.lib.crs
import gws.gis.source
import gws.lib.xmlx as xmlx
import gws.base.ows.client.parseutil as u



def parse(xml: str, bottom_first: bool=False) -> gws.OwsCapabilities:
    """Read WMS capabilities from the GetCapabilities XML.

    Args:
        xml: GetCapabilities XML
        bottom_first: True if layers are listed bottom-first

    Returns:
        The Capabilities object.
    """

    caps_el = xmlx.from_string(xml, gws.XmlOptions(compactWhitespace=True, removeNamespaces=True))
    source_layers = gws.gis.source.check_layers(
        [_layer(el) for el in caps_el.findall('Capability/Layer')],
        revert=bottom_first
    )
    return gws.OwsCapabilities(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        sourceLayers=source_layers,
        version=caps_el.get('version'))


def _layer(layer_el: gws.XmlElement, parent: Optional[gws.SourceLayer] = None) -> gws.SourceLayer:
    sl = gws.SourceLayer()

    sl.isQueryable = layer_el.get('queryable') == '1'
    sl.isVisible = True
    sl.isExpanded = False
    sl.metadata = u.element_metadata(layer_el)
    sl.name = sl.metadata.get('name', '')
    sl.styles = [u.parse_style(e) for e in layer_el.findall('Style')]
    sl.title = sl.metadata.get('title', '')

    # @TODO: support ScaleHint (WMS 1.1)

    smin = layer_el.textof('MinScaleDenominator')
    smax = layer_el.textof('MaxScaleDenominator')
    if smax:
        sl.scaleRange = [u.to_int(smin), u.to_int(smax)]

    wgs_extent = u.wgs_extent(layer_el)
    crs_list = u.supported_crs(layer_el)

    if not parent:
        sl.supportedCrs = crs_list or [gws.lib.crs.WGS84]
        sl.wgsExtent = wgs_extent

    else:
        # OGC 06-042, 7.2.4.8 Inheritance of layer properties

        # Style -> add
        m = {s.name: s for s in sl.styles}
        for s in parent.styles:
            m[s.name] = s
        sl.styles = list(m.values())

        # CRS -> add
        sl.supportedCrs = list(parent.supportedCrs)
        for crs in crs_list:
            if crs not in sl.supportedCrs:
                sl.supportedCrs.append(crs)

        # EX_GeographicBoundingBox -> replace
        sl.wgsExtent = wgs_extent or parent.wgsExtent

        # Dimension -> replace
        # @TODO

        # Attribution -> replace
        sl.metadata.attribution = sl.metadata.attribution or parent.metadata.attribution

        # AuthorityURL -> add
        # @TODO

        # MinScaleDenominator -> replace
        sl.scaleRange = sl.scaleRange or parent.scaleRange

    sl.defaultStyle = u.default_style(sl.styles)
    if sl.defaultStyle:
        sl.legendUrl = sl.defaultStyle.legendUrl

    sl.layers = [_layer(e, sl) for e in layer_el.findall('Layer')]
    sl.isGroup = len(sl.layers) > 0
    sl.isImage = len(sl.layers) == 0

    if not sl.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        sl.isQueryable = False
        sl.isImage = False

    return sl

"""WMS Capabilities parser."""

import gws
import gws.lib.xmlx as xmlx
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


def parse(xml: str) -> core.Caps:
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    sls = gws.gis.source.check_layers(
        _layer(el) for el in caps_el.findall('Capability/Layer'))
    return core.Caps(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        sourceLayers=sls,
        version=caps_el.get('version'))


def _layer(layer_el: gws.IXmlElement, parent: t.Optional[gws.SourceLayer] = None) -> gws.SourceLayer:
    sl = gws.SourceLayer()

    sl.isQueryable = layer_el.get('queryable') == '1'
    sl.isVisible = True
    sl.isExpanded = False
    sl.metadata = u.element_metadata(layer_el)
    sl.name = sl.metadata.get('name', '')
    sl.styles = [u.parse_style(e) for e in layer_el.findall('Style')]
    sl.title = sl.metadata.get('title', '')

    # @TODO: support ScaleHint (WMS 1.1)

    smin = layer_el.text_of('MinScaleDenominator')
    smax = layer_el.text_of('MaxScaleDenominator')
    if smax:
        sl.scaleRange = [u.to_int(smin), u.to_int(smax)]

    # NB don't process axis orders here, this is up to the provider
    bounds, crs_list, wgs_ext = u.bounds_and_crs(layer_el)

    if not parent:
        sl.supportedBounds = bounds
        sl.supportedCrs = crs_list or [gws.gis.crs.WGS84]
        sl.wgsExtent = wgs_ext or gws.gis.crs.WGS84.wgsExtent

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
        sl.wgsExtent = wgs_ext or parent.wgsExtent

        # BoundingBox -> replace
        sl.supportedBounds = bounds or parent.supportedBounds

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

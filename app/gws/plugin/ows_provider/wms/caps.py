"""WMS Capabilities parser."""

import gws
import gws.lib.xmlx as xmlx
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


def parse(xml) -> core.Caps:
    caps_el = xmlx.from_string(xml, compact_whitespace=True, remove_namespaces=True)
    source_layers = gws.gis.source.check_layers(
        _layer(el) for el in caps_el.findall('Capability/Layer'))
    return core.Caps(
        metadata=u.service_metadata(caps_el),
        operations=u.service_operations(caps_el),
        source_layers=source_layers,
        version=caps_el.get('version'))


def _layer(layer_el: gws.IXmlElement, parent: t.Optional[gws.SourceLayer] = None) -> gws.SourceLayer:
    sl = gws.SourceLayer()

    sl.is_queryable = layer_el.get('queryable') == '1'
    sl.is_visible = True
    sl.metadata = u.element_metadata(layer_el)
    sl.name = sl.metadata.get('name', '')
    sl.styles = [u.parse_style(e) for e in layer_el.findall('Style')]
    sl.supported_bounds = u.supported_bounds(layer_el)
    sl.title = sl.metadata.get('title', '')

    if not sl.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        sl.is_queryable = False
        sl.is_image = False

    # @TODO: support ScaleHint (WMS 1.1)

    smin = layer_el.text_of('MinScaleDenominator')
    smax = layer_el.text_of('MaxScaleDenominator')
    if smax:
        sl.scale_range = [u.to_int(smin), u.to_int(smax)]

    # OGC 06-042, 7.2.4.8 Inheritance of layer properties

    if parent:
        crs = set(b.crs for b in sl.supported_bounds)
        for b in parent.supported_bounds:
            if b.crs not in crs:
                sl.supported_bounds.append(b)

        names = set(s.name for s in sl.styles)
        for s in parent.styles:
            if s.name not in names:
                sl.styles.append(s)

        # sl.metadata.extend(parent.metadata)

    sl.layers = [_layer(e, sl) for e in layer_el.findall('Layer')]
    sl.is_group = len(sl.layers) > 0
    sl.is_image = len(sl.layers) == 0

    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    return sl

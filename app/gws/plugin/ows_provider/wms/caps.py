"""WMS Capabilities parser."""

import gws
import gws.lib.xml2 as xml2
import gws.gis.source
import gws.types as t

from .. import core
from .. import parseutil as u


def parse(xml) -> core.Caps:
    root_el = xml2.from_string(xml, strip_ns=True)
    source_layers = gws.gis.source.check_layers(
        _layer(e) for e in xml2.all(root_el, 'Capability Layer'))
    return core.Caps(
        metadata=u.service_metadata(root_el),
        operations=u.service_operations(root_el),
        source_layers=source_layers,
        version=xml2.attr(root_el, 'version'))


def _layer(el: gws.XmlElement, parent: t.Optional[gws.SourceLayer] = None) -> gws.SourceLayer:
    sl = gws.SourceLayer()

    sl.is_queryable = xml2.attr(el, 'queryable') == '1'
    sl.is_visible = True
    sl.metadata = u.element_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.styles = [u.parse_style(e) for e in xml2.all(el, 'Style')]
    sl.supported_bounds = u.supported_bounds(el)
    sl.title = sl.metadata.get('title', '')

    if not sl.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        sl.is_queryable = False
        sl.is_image = False

    # @TODO: support ScaleHint (WMS 1.1)

    smin = xml2.text(el, 'MinScaleDenominator')
    smax = xml2.text(el, 'MaxScaleDenominator')
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

        sl.metadata.extend(parent.metadata)

    sl.layers = [_layer(e, sl) for e in xml2.all(el, 'Layer')]
    sl.is_group = len(sl.layers) > 0
    sl.is_image = len(sl.layers) == 0

    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    return sl

"""WMS Capabilities parser."""

import gws
import gws.lib.ows.parseutil as u
import gws.lib.xml3 as xml3
import gws.types as t

from .. import core


def parse(xml) -> core.Caps:
    root_el = xml3.from_string(xml)
    source_layers = u.enum_source_layers(_layer(e) for e in root_el.all('Capability.Layer'))
    return core.Caps(
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        version=xml3.attr(root_el, 'version'))


def _layer(el, parent=None) -> gws.SourceLayer:
    sl = gws.SourceLayer()

    sl.is_queryable = el.attr('queryable') == '1'
    sl.is_visible = True
    sl.styles = [u.get_style(e) for e in el.all('Style')]

    crs_tags = 'DefaultSRS', 'DefaultCRS', 'SRS', 'CRS', 'OtherSRS'
    extra_crsids = set(e.text for tag in crs_tags for e in el.all(tag))
    sl.supported_bounds = u.get_supported_bounds(el, extra_crsids)

    sl.metadata = u.get_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    if not sl.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        sl.is_queryable = False
        sl.is_image = False

    # @TODO: support ScaleHint (WMS 1.1)

    smin = el.get_text('MinScaleDenominator')
    smax = el.get_text('MaxScaleDenominator')
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

    sl.layers = [_layer(e, sl) for e in el.all('Layer')]
    sl.is_group = len(sl.layers) > 0
    sl.is_image = len(sl.layers) == 0

    sl.default_style = u.default_style(sl.styles)
    if sl.default_style:
        sl.legend_url = sl.default_style.legend_url

    return sl

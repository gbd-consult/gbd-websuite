"""WMS Capabilities parser."""

import gws
import gws.lib.bounds
import gws.lib.gis
import gws.lib.metadata
import gws.lib.ows.parseutil as u
import gws.lib.proj
import gws.lib.xml2
import gws.types as t


class WMSCaps(gws.Data):
    metadata: gws.lib.metadata.Metadata
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml) -> WMSCaps:
    root_el = gws.lib.xml2.from_string(xml)
    source_layers = u.flatten_source_layers(_layer(e) for e in root_el.all('Capability.Layer'))
    return WMSCaps(
        metadata=u.get_service_metadata(root_el),
        operations=u.get_operations(root_el),
        source_layers=source_layers,
        supported_crs=gws.lib.gis.crs_from_source_layers(source_layers),
        version=root_el.attr('version'),
    )


def _layer(el, parent=None) -> gws.lib.gis.SourceLayer:
    sl = gws.lib.gis.SourceLayer()

    sl.supported_bounds = u.get_bounds_list(el)

    if sl.supported_bounds:
        # in addition to specific bounds (above), add crs listed in CRS/SRS tags
        cs = set(b.crs for b in sl.supported_bounds)
        for e in el.all('crs') or el.all('srs'):
            proj = gws.lib.proj.to_proj(e.text)
            if proj and proj.epsg not in cs:
                sl.supported_bounds.append(gws.lib.bounds.transformed_to(sl.supported_bounds[0], proj))
                cs.add(proj.epsg)

    sl.styles = [u.get_style(e) for e in el.all('Style')]
    sl.is_queryable = el.attr('queryable') == '1'
    sl.is_visible = True
    sl.metadata = u.get_metadata(el)
    sl.name = sl.metadata.get('name', '')
    sl.title = sl.metadata.get('title', '')

    if not sl.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        sl.is_queryable = False
        sl.is_image = False

    smin = el.get_text('MinScaleDenominator')
    smax = el.get_text('MaxScaleDenominator')
    if smax:
        sl.scale_range = [u.to_int(smin), u.to_int(smax)]

    # @TODO: support ScaleHint (WMS 1.1)

    sl.layers = [_layer(e, sl) for e in el.all('Layer')]
    sl.is_group = len(sl.layers) > 0
    sl.is_image = len(sl.layers) == 0

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

    sl.supported_crs = [b.crs for b in sl.supported_bounds]

    ds = u.default_style(sl.styles)
    if ds:
        sl.legend_url = ds.legend_url

    return sl

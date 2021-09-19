"""WMS Capabilities parser."""

import gws
import gws.base.ows
import gws.lib.bounds
import gws.lib.gis
import gws.lib.metadata
import gws.lib.ows.parseutil as u
import gws.lib.proj
import gws.lib.xml2
import gws.types as t


class WMSCaps(gws.Data):
    metadata: gws.lib.metadata.Record
    operations: t.List[gws.OwsOperation]
    source_layers: t.List[gws.lib.gis.SourceLayer]
    supported_crs: t.List[gws.Crs]
    version: str


def parse(xml) -> WMSCaps:
    el = gws.lib.xml2.from_string(xml)

    meta = u.get_meta(el.first('Service'))
    meta['contact'] = u.get_meta_contact(el.first('Service.ContactInformation'))

    source_layers = u.flatten_source_layers(_layer(e) for e in el.all('Capability.Layer'))

    return WMSCaps(
        metadata=gws.lib.metadata.Record(meta),
        operations=[gws.OwsOperation(e) for e in u.get_operations(el.first('Capability'))],
        source_layers=source_layers,
        supported_crs=gws.lib.gis.crs_from_source_layers(source_layers),
        version=el.attr('version'),
    )


def _layer(el, parent=None) -> gws.lib.gis.SourceLayer:
    oo = gws.lib.gis.SourceLayer()

    oo.supported_bounds = u.get_bounds_list(el)

    if oo.supported_bounds:
        # in addition to specific bounds (above), add crs listed in CRS/SRS tags
        cs = set(b.crs for b in oo.supported_bounds)
        for e in el.all('crs') or el.all('srs'):
            proj = gws.lib.proj.as_proj(e.text)
            if proj and proj.epsg not in cs:
                oo.supported_bounds.append(gws.lib.bounds.transformed_to(oo.supported_bounds[0], proj))
                cs.add(proj.epsg)

    oo.styles = [u.get_style(e) for e in el.all('Style')]
    oo.is_queryable = el.attr('queryable') == '1'
    oo.is_visible = True
    oo.metadata = gws.lib.metadata.Record(u.get_meta(el))
    oo.name = oo.metadata.name or ''
    oo.title = oo.metadata.title or ''

    if not oo.name:
        # some folks have unnamed layers in their caps
        # we can't render or query them
        oo.is_queryable = False
        oo.is_image = False

    smin = el.get_text('MinScaleDenominator')
    smax = el.get_text('MaxScaleDenominator')
    if smax:
        oo.scale_range = [u.as_int(smin), u.as_int(smax)]

    # @TODO: support ScaleHint (WMS 1.1)

    oo.layers = [_layer(e, oo) for e in el.all('Layer')]
    oo.is_group = len(oo.layers) > 0
    oo.is_image = len(oo.layers) == 0

    # OGC 06-042, 7.2.4.8 Inheritance of layer properties

    if parent:
        crs = set(b.crs for b in oo.supported_bounds)
        for b in parent.supported_bounds:
            if b.crs not in crs:
                oo.supported_bounds.append(b)

        names = set(s.name for s in oo.styles)
        for s in parent.styles:
            if s.name not in names:
                oo.styles.append(s)

        oo.metadata = gws.lib.metadata.merge(parent.metadata, oo.metadata)

    oo.supported_crs = [b.crs for b in oo.supported_bounds]

    ds = u.default_style(oo.styles)
    if ds:
        oo.legend_url = ds.legend_url

    return oo

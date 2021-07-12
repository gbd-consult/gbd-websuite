"""WMS Capabilities parser."""

import gws
import gws.types as t
import gws.lib.metadata
import gws.base.ows.provider.parseutil as u
import gws.lib.bounds
import gws.lib.proj
import gws.lib.source
import gws.lib.xml2


def parse(prov, xml):
    el = gws.lib.xml2.from_string(xml)

    prov.type = 'WMS'
    prov.meta = gws.MetaData(u.get_meta(el.first('Service')))
    prov.meta.contact = gws.MetaContact(u.get_meta_contact(el.first('Service.ContactInformation')))
    prov.version = el.attr('version')
    prov.operations = u.get_operations(el.first('Capability'))
    prov.source_layers = u.flatten_source_layers(_layer(e) for e in el.all('Capability.Layer'))
    prov.supported_crs = gws.lib.source.crs_from_layers(prov.source_layers)


def _layer(el, parent=None):
    oo = gws.SourceLayer()

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
    oo.meta = gws.MetaData(u.get_meta(el))
    oo.name = oo.meta.name
    oo.title = oo.meta.title

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
        cs = set(b.crs for b in oo.supported_bounds)
        for b in parent.supported_bounds:
            if b.crs not in cs:
                oo.supported_bounds.append(b)

        cs = set(b.crs for b in oo.styles)
        for s in parent.styles:
            if s.name not in cs:
                oo.styles.append(s)

        oo.meta = gws.merge(parent.meta, oo.meta)

    oo.supported_crs = [b.crs for b in oo.supported_bounds]

    ds = u.default_style(oo.styles)
    if ds:
        oo.legend = ds.legend

    return oo

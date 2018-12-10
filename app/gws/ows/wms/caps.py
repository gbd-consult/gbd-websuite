import re

import gws
import gws.types as t
import gws.tools.xml3
import gws.tools.net
import gws.gis.proj

import gws.ows.parseutil as u
import gws.ows.request
import gws.ows.error

from . import types


def parse(srv: t.ServiceInterface, xml):
    el = gws.tools.xml3.from_string(xml)

    srv.meta = t.MetaData(u.get_meta(el.first('Service')))
    srv.meta.contact = t.MetaContact(u.get_meta_contact(el.first('Service.ContactInformation')))
    srv.meta.url = u.get_url(el.first('Service.OnlineResource'))
    srv.version = el.attr('version')
    srv.operations = u.get_operations(el.first('Capability'))
    srv.layers = u.flatten_source_layers(_layer(e) for e in el.all('Capability.Layer'))
    srv.supported_crs = u.crs_from_layers(srv.layers)


def _layer(el):
    oo = types.SourceLayer()

    oo.supported_crs = [e.text for e in el.all('CRS')] + [e.text for e in el.all('SRS')]
    oo.extents = u.get_extents(el)

    oo.styles = [u.get_style(e) for e in el.all('Style')]
    ds = u.default_style(oo.styles)
    if ds:
        oo.legend = ds.legend

    oo.layers = [_layer(e) for e in el.all('Layer')]

    oo.is_group = len(oo.layers) > 0
    oo.is_image = len(oo.layers) == 0
    oo.is_queryable = el.attr('queryable') == '1'
    oo.is_visible = True

    oo.meta = t.MetaData(u.get_meta(el))
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

    return oo

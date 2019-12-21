import gws
import gws.gis.source
import gws.gis.util

from . import provider


def configure_wms(target: gws.Object, **filter_args):
    target.url = target.var('url')

    target.provider = gws.gis.util.shared_ows_provider(provider.Object, target, target.config)
    target.axis = target.var('axis')
    target.source_layers = gws.gis.source.filter_layers(
        target.provider.source_layers,
        target.var('sourceLayers'),
        **filter_args)

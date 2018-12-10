import re
from argh import arg

import gws
import gws.types as t
import gws.tools.json2
import gws.tools.clihelpers

import gws.qgis
import gws.ows.wms
import gws.ows.wfs
import gws.ows.wmts

COMMAND = 'ows'


def _get_service(type, src):
    is_url = re.match(r'^https?:', src)

    r = gws.RootObject()

    if not is_url and src.endswith('.qgs'):
        return r.create_object(gws.qgis.Service, t.Config({'path': src}))

    if is_url:
        cfg = {'url': src}
    else:
        with open(src) as fp:
            cfg = {'xml': fp.read()}

    if type.lower() == 'wms':
        return r.create_object(gws.ows.wms.Service, cfg)
    if type.lower() == 'wfs':
        return r.create_object(gws.ows.wfs.Service, cfg)
    if type.lower() == 'wmts':
        return r.create_object(gws.ows.wmts.Service, cfg)


@arg('--type', help='service type')
@arg('--src', help='service url or path')
def caps(type=None, src=None):
    """Query the capabilities of a service"""

    srv = _get_service(type, src)
    if srv:
        print(gws.tools.json2.to_tagged_string(srv, pretty=True))


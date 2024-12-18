"""CLI utilty for OWS services"""

from typing import Optional, cast

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.importer
import gws.lib.jsonx


from . import request


class CapsParams(gws.CliParams):
    src: str
    """service URL or an XML file name"""
    type: str = ''
    """service type, e.g. WMS"""
    out: str = ''
    """output filename"""


class Object(gws.Node):

    @gws.ext.command.cli('owsCaps')
    def caps(self, p: CapsParams):
        """Print the capabilities of a service in JSON format"""

        protocol = None

        if p.type:
            protocol = p.type.lower()
        else:
            u = p.src.lower()
            for s in ('wms', 'wmts', 'wfs'):
                if s in u:
                    protocol = s
                    break

        if not protocol:
            raise gws.Error('unknown service')

        if p.src.startswith(('http:', 'https:')):
            xml = request.get_text(request.Args(
                url=p.src,
                protocol=cast(gws.OwsProtocol, protocol.upper()),
                verb=gws.OwsVerb.GetCapabilities))
        else:
            xml = gws.u.read_file(p.src)

        mod = gws.lib.importer.import_from_path(f'gws/plugin/ows_client/{protocol}/caps.py')
        res = mod.parse(xml)

        js = gws.lib.jsonx.to_pretty_string(res, default=_caps_json)

        if p.out:
            gws.u.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)


def _caps_json(x):
    if isinstance(x, gws.lib.crs.Crs):
        return x.epsg
    if isinstance(x, gws.base.shape.Shape):
        return x.to_geojson()
    try:
        return vars(x)
    except TypeError:
        return repr(x)

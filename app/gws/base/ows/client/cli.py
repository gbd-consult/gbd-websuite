"""CLI command for OWS services."""

from typing import Optional, cast

import gws
import gws.lib.shape
import gws.lib.crs
import gws.lib.dynimport
import gws.lib.jsonx


from . import request


class CapsParams(gws.CliParams):
    """Parameters for the ``owsCaps`` command."""

    src: str
    """Service URL or XML file name."""
    type: str = ''
    """Service type, e.g. WMS. If omitted, it is guessed from ``src``."""
    out: str = ''
    """Output file name. If omitted, the result is printed."""


class Object(gws.Node):
    """CLI commands for OWS services."""

    @gws.ext.command.cli('owsCaps')
    def caps(self, p: CapsParams):
        """Print the capabilities of a service in JSON format."""

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

        mod = gws.lib.dynimport.import_from_path(f'gws/plugin/ows_client/{protocol}/caps.py')
        res = mod.parse(xml)

        js = gws.lib.jsonx.to_pretty_string(res, default=_caps_json)

        if p.out:
            gws.u.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)


def _caps_json(x):
    """Convert an object that is not JSON-serializable for the JSON output."""
    if isinstance(x, gws.lib.crs.Object):
        return x.epsg
    if isinstance(x, gws.lib.shape.Shape):
        return x.to_geojson()
    try:
        return vars(x)
    except TypeError:
        return repr(x)

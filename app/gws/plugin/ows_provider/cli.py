"""CLI utilty for OWS services"""

import gws
import gws.lib.json2
import gws.gis.ows
import gws.types as t


class CapsParams(gws.CliParams):
    src: str  #: service URL or an XML file name
    type: str = ''  #: service type, e.g. WMS
    out: str = ''  #: output filename


@gws.ext.Object('cli.ows')
class Object(gws.Node):

    @gws.ext.command('cli.ows.caps')
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
            xml = gws.gis.ows.request.get_text(
                p.src,
                protocol=t.cast(gws.OwsProtocol, protocol.upper()),
                verb=gws.OwsVerb.GetCapabilities)
        else:
            xml = gws.read_file(p.src)

        mod = gws.import_from_path(f'gws/plugin/ows_provider/{protocol}/caps.py')
        res = mod.parse(xml)

        def js(x):
            if isinstance(x, gws.IMetadata):
                return vars(x.values)
            if isinstance(x, gws.ICrs):
                return x.epsg
            try:
                return vars(x)
            except TypeError:
                return repr(x)

        js = gws.lib.json2.to_pretty_string(res, default=js)

        if p.out:
            gws.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)

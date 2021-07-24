"""CLI utilty for OWS services"""

import gws
import gws.lib.json2
import gws.lib.ows


class CapsParams:
    url: gws.Url  #: service URL or an XML file name
    service: str = ''  #: service name, e.g. WMS
    out: str = ''  #: output filename


@gws.ext.Object('cli.ows')
class Cli:

    @gws.ext.command('cli.ows.caps')
    def caps(self, p: CapsParams):
        """Print the capabilities of a service in JSON format"""

        service = None

        if p.service:
            service = p.service.lower()
        else:
            u = p.url.lower()
            for s in ('wms', 'wmts', 'wfs'):
                if s in u:
                    service = s
                    break

        if not service:
            raise gws.Error('cannot guess the service name')

        gws.log.info(f'using service {service}...')

        if p.url.startswith(('http:', 'https:')):
            xml = gws.lib.ows.request.get_text(p.url, service=service, verb='GetCapabilities')
        else:
            xml = gws.read_file(p.url)

        mod = gws.import_from_path(
            f'{gws.APP_DIR}/gws/plugin/ows_provider/{service}/caps.py',
            f'gws.plugin.ows_provider.{service}.caps'
        )

        res = mod.parse(xml)
        js = gws.lib.json2.to_pretty_string(res)

        if p.out:
            gws.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)

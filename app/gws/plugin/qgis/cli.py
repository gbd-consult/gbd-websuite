"""CLI utilty for OWS services"""

import gws
import gws.lib.json2
import gws.gis.ows
import gws.types as t


class CapsParams(gws.CliParams):
    path: str  #: path to a qgis file
    out: str = ''  #: output filename


@gws.ext.Object('cli.qgis')
class Object(gws.Node):

    @gws.ext.command('cli.qgis.caps')
    def caps(self, p: CapsParams):
        """Print the capabilities of a document in JSON format"""

        xml = gws.read_file(p.path)
        mod = gws.import_from_path(f'gws/plugin/qgis/caps.py')
        res = mod.parse(xml)

        js = gws.lib.json2.to_pretty_string(res, default=_caps_json, ascii=False)

        if p.out:
            gws.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)


def _caps_json(x):
    if isinstance(x, gws.IMetadata):
        return vars(x.values)
    if isinstance(x, gws.ICrs):
        return x.epsg
    if isinstance(x, gws.IShape):
        return x.to_geojson()
    try:
        return vars(x)
    except TypeError:
        return repr(x)

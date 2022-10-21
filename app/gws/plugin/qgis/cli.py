"""CLI utilty for qgis"""

import gws
import gws.base.shape
import gws.gis.crs
import gws.lib.importer
import gws.lib.jsonx


class CapsParams(gws.CliParams):
    path: str  #: path to a qgis file
    out: str = ''  #: output filename


class Object(gws.Node):

    @gws.ext.command.cli('qgisCaps')
    def caps(self, p: CapsParams):
        """Print the capabilities of a document in JSON format"""

        xml = gws.read_file(p.path)

        mod = gws.lib.importer.import_from_path(f'gws/plugin/qgis/caps.py')
        res = mod.parse(xml)

        js = gws.lib.jsonx.to_pretty_string(res, default=_caps_json)

        if p.out:
            gws.write_file(p.out, js)
            gws.log.info(f'saved to {p.out!r}')
        else:
            print(js)


def _caps_json(x):
    if isinstance(x, gws.gis.crs.Crs):
        return x.epsg
    if isinstance(x, gws.base.shape.Shape):
        return x.to_geojson()
    try:
        return vars(x)
    except TypeError:
        return repr(x)

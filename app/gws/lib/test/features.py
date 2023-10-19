import datetime

import gws
import gws.base.feature
import gws.lib.osx
import gws.lib.jsonx

from . import util


def make(table_name, geom_type, column_defs, crs, start_xy, num_rows, num_cols, gap):
    features = []

    sx, sy = start_xy

    for r in range(num_rows):
        for c in range(num_cols):
            uid = r * num_cols + (c + 1)
            atts = []

            for name, typ in column_defs.items():
                val = ''
                if typ == 'int':
                    val = uid * 100
                if typ == 'float':
                    val = uid * 200.0
                if typ in ('varchar', 'text'):
                    val = f"{name}/{uid}"
                if typ == 'date':
                    val = datetime.datetime(2019, 1, 1) + datetime.timedelta(days=uid - 1)
                atts[name] = val

            x = sx + c * gap
            y = sy + r * gap

            geom = None

            if geom_type == 'point':
                geom = {
                    'type': 'Point',
                    'coordinates': [x, y]
                }

            if geom_type == 'square':
                w = h = gap / 2
                geom = {
                    'type': 'Polygon',
                    'coordinates': [[
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h],
                        [x, y],
                    ]]
                }

            features.append(gws.Data(
                uid=uid,
                attributes=atts,
                shape={'crs': crs, 'geometry': geom} if geom else None
            ))

    return features


def make_geojson(path, geom_type, columns, crs, xy, rows, cols, gap):
    name = gws.lib.osx.parse_path(path)['name']
    features = make(name, geom_type, columns, crs, xy, rows, cols, gap)
    text = gws.lib.jsonx.to_pretty_string({
        'type': 'FeatureCollection',
        'crs': {'type': 'name', 'properties': {'name': crs}},
        'features': [f.to_geojson() for f in features],
    })
    util.write_file_if_changed(path, text)

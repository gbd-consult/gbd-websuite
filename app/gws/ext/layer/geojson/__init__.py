import gws.common.layer
import gws.gis.shape
import gws.gis.feature
import gws.gis.proj
import gws.common.db
import gws.tools.json2

import gws.types as t


class Config(gws.common.layer.VectorConfig):
    """GeoJson layer"""

    path: t.FilePath  #: geojson file
    keyProp: str = 'id'  #: property name for unique ids


class Object(gws.common.layer.Vector):
    def __init__(self):
        super().__init__()
        self.path = ''
        self.features: t.List[t.IFeature] = []

    def configure(self):
        super().configure()

        self.path = self.var('path')
        js = gws.tools.json2.from_path(self.path)
        crs = gws.gis.proj.as_epsg(_get_crs(js) or 'EPSG:4326')
        for f in js['features']:
            f = gws.gis.feature.from_geojson(f, crs, self.var('keyProp'))
            self.features.append(f.transform(self.crs))

    def get_features(self, bbox, limit=0):
        shape = gws.gis.shape.from_bbox(bbox, self.crs)
        fs = [f for f in self.features if f.shape.intersects(shape)]
        if limit:
            fs = fs[:limit]
        return [self.connect_feature(f) for f in fs]


def _get_crs(js):
    crs = js.get('crs')
    if not crs:
        return
    if crs['type'] == 'name':
        return crs['properties']['name']
    if crs['type'] == 'epsg':
        return 'EPSG:' + str(crs['properties']['code'])
    raise ValueError(f'unsupported crs type {crs["type"]!r}')

import gws
import gws.types as t
import gws.base.db
import gws.base.layer
import gws.lib.feature
import gws.lib.proj
import gws.lib.shape
import gws.lib.json2


class Config(gws.base.layer.VectorConfig):
    """GeoJson layer"""

    path: gws.FilePath  #: geojson file
    keyProp: str = 'id'  #: property name for unique ids


class Object(gws.base.layer.Vector):
    def configure(self):
        

        self.path = self.var('path')
        js = gws.lib.json2.from_path(self.path)
        self.own_crs = gws.lib.proj.as_epsg(_get_crs(js) or gws.EPSG_4326)
        self.features = [
            gws.lib.feature.from_geojson(f, self.crs, self.var('keyProp'))
            for f in js['features']]

    def get_features(self, bounds, limit=0):
        shape = gws.lib.shape.from_bounds(bounds).transformed_to(self.own_crs)
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

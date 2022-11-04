"""GeoJSON layer"""

import gws
import gws.base.layer
import gws.gis.crs
import gws.lib.jsonx
import gws.base.feature
import gws.base.shape
import gws.gis.ows
import gws.types as t


@gws.ext.config.layer('geojson')
class Config(gws.base.layer.vector.Config):
    """GeoJson layer"""

    path: gws.FilePath 
    """geojson file"""
    keyName: str = 'id' 
    """property name for unique ids"""


@gws.ext.object.layer('geojson')
class Object(gws.base.layer.vector.Object, gws.IOwsClient):
    path: str
    source_crs: gws.ICrs
    features: t.List[gws.IFeature]

    def configure_source(self):
        self.path = self.var('path')
        js = gws.lib.jsonx.from_path(self.path)
        self.source_crs = self._get_crs(js) or self.map.crs
        self.features = [
            gws.base.feature.from_geojson(f, self.crs, self.var('keyName'))
            for f in js['features']]

    def get_features(self, bounds, limit=0):
        shape = gws.base.shape.from_bounds(bounds).transformed_to(self.source_crs)
        fs = [f for f in self.features if f.shape.intersects(shape)]
        if limit:
            fs = fs[:limit]
        return [f.connect_to(self) for f in fs]

    def _get_crs(self, js):
        p = js.get('crs')
        if not p:
            return

        if p.get('type') == 'name':
            crs = gws.gis.crs.get(gws.get(p, 'properties.name'))
            if crs:
                return crs

        raise gws.Error(f'unsupported geojson crs format')

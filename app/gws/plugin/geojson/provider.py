"""GeoJson provder."""

import gws
import gws.gis.crs
import gws.base.shape
import gws.lib.jsonx


class Config(gws.Config):
    path: gws.FilePath
    """path to a geojson file"""


class Object(gws.Node):
    path: str

    _featureData = None

    def configure(self):
        self.path = self.cfg('path')

    def post_configure(self):
        self._featureData = None

    def feature_data(self) -> list[gws.FeatureData]:
        if self._featureData is None:
            self._featureData = self._load()
        return self._featureData

    def _load(self):
        js = gws.lib.jsonx.from_path(self.path)

        crs = gws.gis.crs.WGS84
        if 'crs' in js:
            # https://geojson.org/geojson-spec#named-crs
            crs = gws.gis.crs.get(js['crs']['properties']['name'])

        fds = []

        for f in js.get('features', []):
            fd = gws.FeatureData(attributes=f.get('properties', {}))
            if f.get('geometry'):
                fd.shape = gws.base.shape.from_geojson(f['geometry'], crs)
            fds.append(fd)

        return fds


##


def get_for(obj: gws.INode) -> Object:
    p = obj.cfg('provider')
    if p:
        return obj.root.create_shared(Object, p)
    p = obj.cfg('_defaultProvider')
    if p:
        return p
    raise gws.Error(f'no provider found for {obj!r}')

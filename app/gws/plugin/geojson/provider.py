"""GeoJson provder."""

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.jsonx


class Config(gws.Config):
    """Configuration for GeoJson provider."""

    path: gws.FilePath
    """path to a geojson file"""

    path: gws.FilePath
    """Path to a geojson file."""


class Object(gws.Node):
    path: str

    _records = None

    def configure(self):
        self.path = self.cfg('path')

    def post_configure(self):
        self._records = None

    def get_records(self) -> list[gws.FeatureRecord]:
        if self._records is None:
            self._records = self._load()
        return self._records

    def _load(self):
        js = gws.lib.jsonx.from_path(self.path)

        crs = gws.lib.crs.WGS84
        if 'crs' in js:
            # https://geojson.org/geojson-spec#named-crs
            crs = gws.lib.crs.get(js['crs']['properties']['name'])

        records = []

        for n, f in enumerate(js.get('features', []), 1):
            p = f.get('properties', {})
            rec = gws.FeatureRecord(attributes=p)
            if f.get('geometry'):
                rec.shape = gws.base.shape.from_geojson(f['geometry'], crs)
            rec.uid = p.get('id') or p.get('uid') or p.get('fid') or p.get('sid') or n
            records.append(rec)

        return records

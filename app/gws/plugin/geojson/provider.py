"""GeoJson provder."""

from typing import Optional, cast

import gws
import gws.config.util
import gws.gis.crs
import gws.base.shape
import gws.lib.jsonx



class Config(gws.Config):
    path: gws.FilePath
    """path to a geojson file"""


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

        crs = gws.gis.crs.WGS84
        if 'crs' in js:
            # https://geojson.org/geojson-spec#named-crs
            crs = gws.gis.crs.get(js['crs']['properties']['name'])

        records = []

        for n, f in enumerate(js.get('features', []), 1):
            p = f.get('properties', {})
            rec = gws.FeatureRecord(attributes=p)
            if f.get('geometry'):
                rec.shape = gws.base.shape.from_geojson(f['geometry'], crs)
            rec.uid = p.get('id') or p.get('uid') or p.get('fid') or p.get('sid') or n
            records.append(rec)

        return records


##


def get_for(obj: gws.Node) -> Object:
    return cast(Object, gws.config.util.get_provider(Object, obj))

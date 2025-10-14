"""GeoJSON provder."""

from typing import Optional
import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.bounds
import gws.lib.jsonx


class Config(gws.Config):
    """Configuration for GeoJSON provider."""

    path: gws.FilePath
    """path to a GeoJSON file"""


class Object(gws.Node):
    path: str
    _records: list[gws.FeatureRecord]

    def __getstate__(self):
        return gws.u.omit(vars(self), '_records')

    def configure(self):
        self.path = self.cfg('path')

    def load_records(self):
        if getattr(self, '_records', None) is None:
            self._records = self._load()
        return self._records

    def get_records(self, search: gws.SearchQuery) -> list[gws.FeatureRecord]:
        shape = None
        
        if search.shape:
            shape = search.shape
            if search.tolerance:
                tol_value, tol_unit = search.tolerance
                if tol_unit == gws.Uom.px:
                    tol_value *= search.resolution
                shape = shape.tolerance_polygon(tol_value)
        elif search.bounds:
            shape = gws.base.shape.from_bounds(search.bounds)

        return [rec for rec in self.load_records() if self._record_matches(rec, search, shape)]

    def _record_matches(self, rec: gws.FeatureRecord, search: gws.SearchQuery, shape: Optional[gws.Shape]) -> bool:
        if shape:
            if not rec.shape or not rec.shape.intersects(shape):
                return False

        if search.keyword:
            if all(search.keyword.lower() not in str(v).lower() for v in rec.attributes.values()):
                return False

        if search.uids and rec.uid not in search.uids:
            return False

        return True

    def _load(self):
        js = gws.lib.jsonx.from_path(self.path)

        crs = gws.lib.crs.WGS84
        if 'crs' in js:
            # https://geojson.org/geojson-spec#named-crs
            crs = gws.lib.crs.require(js['crs']['properties']['name'])

        records = []

        for n, f in enumerate(js.get('features', []), 1):
            p = f.get('properties', {})
            rec = gws.FeatureRecord(attributes=p)
            if f.get('geometry'):
                rec.shape = gws.base.shape.from_geojson(f['geometry'], crs)
            rec.uid = p.get('id') or p.get('uid') or p.get('fid') or p.get('sid') or str(n) or ''
            records.append(rec)

        return records

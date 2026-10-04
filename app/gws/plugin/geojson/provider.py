"""GeoJSON provider."""

from typing import Optional
import gws
import gws.lib.shape
import gws.lib.crs
import gws.lib.bounds
import gws.lib.jsonx


class Config(gws.Config):
    """Access to a GeoJSON file."""

    path: gws.FilePath
    """Path to a GeoJSON file."""


class Object(gws.Node):
    """GeoJSON file provider.

    Reads the features of a GeoJSON file into feature records and selects
    the records that match a search.
    """

    path: str
    """Path to the GeoJSON file."""
    _records: list[gws.FeatureRecord]

    def __getstate__(self):
        """Omit the loaded records when pickling."""
        return gws.u.omit(vars(self), '_records')

    def configure(self):
        self.path = self.cfg('path')

    def cache_hash(self):
        """Return a hash that identifies the data of the provider.

        Returns:
            A hash of the file path.
        """
        return gws.u.sha256([self.path])

    def load_records(self):
        """Return all records of the file.

        The file is read on the first call.

        Returns:
            The feature records.
        """
        if getattr(self, '_records', None) is None:
            self._records = self._load()
        return self._records

    def get_records(self, search: gws.SearchQuery) -> list[gws.FeatureRecord]:
        """Return the records that match a search.

        A record matches if its shape intersects the search shape, extended by
        the tolerance, or, without a shape, the search bounds; if one of its
        attribute values contains the keyword, ignoring case; and if its uid
        is one of the search uids. Criteria not set in the search are ignored.

        Args:
            search: The search query.

        Returns:
            The matching records.
        """
        shape = None
        
        if search.shape:
            shape = search.shape
            if search.tolerance:
                tol_value, tol_unit = search.tolerance
                if tol_unit == gws.Uom.px:
                    tol_value *= search.resolution
                shape = shape.tolerance_polygon(tol_value)
        elif search.bounds:
            shape = gws.lib.shape.from_bounds(search.bounds)

        return [rec for rec in self.load_records() if self._record_matches(rec, search, shape)]

    def _record_matches(self, rec: gws.FeatureRecord, search: gws.SearchQuery, shape: Optional[gws.Shape]) -> bool:
        """Check if a record matches the search criteria."""
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
        """Read the GeoJSON file into feature records."""
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
                rec.shape = gws.lib.shape.from_geojson(f['geometry'], crs)
            rec.uid = p.get('id') or p.get('uid') or p.get('fid') or p.get('sid') or str(n) or ''
            records.append(rec)

        return records

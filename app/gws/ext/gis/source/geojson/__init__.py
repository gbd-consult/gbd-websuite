import gws
import gws.config
import gws.tools.json2 as json2
import gws.gis.proj
import gws.gis.feature
import gws.gis.source
import gws.gis.shape
import gws.types as t


class Config(t.WithTypeAndAccess):
    """GeoJson source"""

    #: CRS for this source
    crs: t.Optional[t.crsref]
    #: path to a geojson file
    path: t.filepath
    #: text search attribute
    searchAttribute: t.Optional[str]


class Object(gws.gis.source.Base):
    search_attr: str = None

    def configure(self):
        super().configure()
        self.search_attr = self.var('searchAttribute')

    def _get_all(self):
        d = json2.from_path(self.var('path'))

        return [
            gws.gis.feature.from_geojs(p, self.var('crs'))
            for p in d.get('features', [])
        ]

    def _write(self, features):
        js = {
            'type': 'FeatureCollection',
            'features': [f.to_geojs() for f in features]
        }
        json2.to_path(self.var('path'), js, pretty=True)

    def get_features(self, keyword, shape: gws.gis.shape.Shape, sort=None, limit=None):

        fs = self._get_all()

        if keyword and self.search_attr:
            kw = keyword.lower()
            fs = [f for f in fs if kw in f.attributes.get(self.search_attr, '').lower()]

        if shape:
            fs = [f for f in fs if f.shape.geo.intersects(shape.geo)]

        return fs

    def modify_features(self, operation, fp: t.List[t.FeatureProps]):
        fs = self._get_all()
        existing = set(f.uid for f in fs)

        if operation == 'add':
            # @TODO: merge 'data' from params with feature defaults

            add = []
            for p in fp:
                f = gws.gis.feature.from_props(p)
                if f.uid in existing:
                    continue
                if not f.uid:
                    f.uid = gws.random_string(16)
                add.append(f)

            if add:
                self._write(fs + add)

            return

        if operation == 'delete':
            delete = set()
            for p in fp:
                if p.uid in existing:
                    delete.add(p.uid)

            if delete:
                fs = [f for f in fs if f.uid not in delete]
                self._write(fs)

            return

        if operation == 'update':
            # @TODO: merge 'data' from params with feature data
            update = []
            update_ids = set()

            for p in fp:
                f = gws.gis.feature.from_props(p)
                if f.uid not in existing or f.uid in update_ids:
                    continue
                update.append(f)
                update_ids.add(f.uid)

            if update:
                fs = [f for f in fs if f.uid not in update_ids]
                self._write(fs + update)

            return

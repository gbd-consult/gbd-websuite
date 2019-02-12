import gws.gis.layer
import gws.gis.shape
import gws.tools.misc as misc

import gws.types as t


class Config(gws.gis.layer.VectorConfig):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: t.SqlTableConfig  #: sql table configuration


class Object(gws.gis.layer.Vector):
    def configure(self):
        super().configure()

        self.db: t.DbProviderObject = None
        s = self.var('db')
        if s:
            self.db = self.root.find('gws.ext.db.provider', s)
        else:
            self.db = self.root.find_first('gws.ext.db.provider')

        if not self.db:
            raise gws.Error(f'{self.uid}: db provider not found')

        self.table = self.var('table')
        with self.db.connect() as conn:
            self.crs = conn.crs_for_column(self.table.name, self.table.geometryColumn)
            self.geometry_type = conn.geometry_type_for_column(self.table.name, self.table.geometryColumn)

        self.add_child('gws.ext.search.provider.sql', t.Config({
            'db': self.db.uid,
            'table': self.table,
        }))

    @property
    def props(self):
        return gws.extend(super().props, {
            'type': 'vector',
            'geometryType': self.geometry_type.upper(),
        })

    def get_features(self, bbox):
        shape = gws.gis.shape.from_bbox(bbox, self.crs)

        fs = self.db.select(t.SelectArgs({
            'table': self.var('table'),
            'shape': shape,
        }))

        return self._format_features(fs)

    def update_features(self, features: t.List[t.FeatureProps]):
        recs = []

        for f in features:
            rec = _noempty(f.attributes)
            if gws.get(f, 'shape'):
                rec[self.var('table').geometryColumn] = gws.gis.shape.from_props(f.shape)
            rec[self.var('table').keyColumn] = self._get_id(f)
            recs.append(rec)

        ids = self.db.update(self.table, recs)
        return self._get_by_ids(ids)

    def add_features(self, features: t.List[t.FeatureProps]):
        recs = []

        for f in features:
            rec = _noempty(f.attributes)
            if gws.get(f, 'shape'):
                rec[self.var('table').geometryColumn] = gws.gis.shape.from_props(f.shape)
            recs.append(rec)

        ids = self.db.insert(self.table, recs)
        return self._get_by_ids(ids)

    def delete_features(self, features: t.List[t.FeatureProps]):
        recs = []

        for f in features:
            rec = {}
            rec[self.var('table').keyColumn] = self._get_id(f)
            recs.append(rec)

        ids = self.db.delete(self.table, recs)
        return []

    def render_svg(self, bbox, dpi, scale, rotation, style):
        features = self.get_features(bbox)
        for f in features:
            f.set_default_style(style)
        return [f.to_svg(bbox, dpi, scale, rotation) for f in features]

    def _get_by_ids(self, ids):
        fs = self.db.select(t.SelectArgs({
            'table': self.var('table'),
            'ids': list(ids),
        }))

        return self._format_features(fs)

    def _format_features(self, fs):
        for f in fs:
            f.apply_format(self.feature_format)
            f.uid = misc.sha256(self.uid) + '_' + str(f.attributes.get(self.var('table').keyColumn))
        return fs

    def _get_id(self, f):
        return f.uid.split('_')[1]


def _noempty(d):
    if not d:
        return {}
    return {k: v for k, v in d.items() if v != ''}

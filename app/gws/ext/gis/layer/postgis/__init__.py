import gws.types as t
import gws.gis.layer
import gws.gis.shape


class Config(gws.gis.layer.VectorConfig):
    """postgis layer"""

    db: t.Optional[str]  #: database provider uid
    table: t.SqlTableConfig  #: sql table configuration


class Object(gws.gis.layer.Vector):
    def configure(self):
        super().configure()
        self.db: t.DbProviderObject = self.root.find('gws.ext.db.provider', self.var('db'))
        if not self.db:
            raise gws.Error(f'{self.uid}: db provider not found')
        self.table = self.var('table')
        with self.db.connect() as conn:
            self.crs = conn.crs_for_column(self.table.name, self.table.geometryColumn)
            self.geometry_type = conn.geometry_type_for_column(self.table.name, self.table.geometryColumn)

    @property
    def props(self) -> gws.gis.layer.VectorProps:
        return gws.extend(super().props, {
            'type': 'vector',
        })

    def get_features(self, bbox):
        shape = gws.gis.shape.from_bbox(bbox, self.crs)

        fs = self.db.select(t.SelectArgs({
            'table': self.var('table'),
            'shape': shape,
        }))

        return self._format_features(fs)

    def update_features(self, features: t.List[t.FeatureProps]):
        with self.db.connect() as conn:
            with conn.transaction():
                for f in features:
                    conn.update(self.table.name, self.table.keyColumn, f.uid, self._prepare_for_db(f))

        return self._get_by_ids(f.uid for f in features)

    def add_features(self, features: t.List[t.FeatureProps]):
        ids = []

        with self.db.connect() as conn:
            with conn.transaction():
                for f in features:
                    ids.append(conn.insert_one(self.table.name, self.table.keyColumn, self._prepare_for_db(f)))

        return self._get_by_ids(ids)

    def render_svg(self, bbox, dpi, scale, rotation, style):
        features = self.get_features(bbox)
        for f in features:
            f.set_default_style(style)
        return [f.to_svg(bbox, dpi, scale, rotation) for f in features]

    def _prepare_for_db(self, f):
        data = f.attributes or {}

        if gws.get(f, 'shape'):
            shape = gws.gis.shape.from_props(f.shape)
            shape.transform(self.crs)
            ph = 'ST_SetSRID(%s::geometry,%s)'
            if self.geometry_type.startswith('MULTI'):
                ph = f'ST_Multi({ph})'
            data[self.table.geometryColumn] = [ph, shape.wkb_hex, shape.crs_code]

        return data

    def _get_by_ids(self, ids):
        fs = self.db.select(t.SelectArgs({
            'table': self.var('table'),
            'ids': list(ids),
        }))

        return self._format_features(fs)

    def _format_features(self, fs):
        for f in fs:
            f.apply_format(self.feature_format)
            f.uid = str(f.attributes.get(self.var('table').keyColumn))

        return fs

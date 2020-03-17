import gws.common.layer
import gws.gis.shape
import gws.gis.feature
import gws.common.db
import gws.common.search.provider
import gws.gis.extent
import gws.gis.shape

import gws.types as t

from . import provider


class Config(gws.common.layer.VectorConfig):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: gws.common.db.SqlTableConfig  #: sql table configuration


class Object(gws.common.layer.Vector):

    def configure(self):
        super().configure()

        self.is_editable = True

        self.provider = t.cast(provider.Object, gws.common.db.require_provider(self, provider.Object))
        self.table = self.provider.configure_table(self.var('table'))

        if not self.data_model:
            p = self.provider.table_data_model_config(self.table)
            if p:
                self.data_model = self.add_child('gws.common.model', p)

    @property
    def own_bounds(self):
        if not self.table.geometry_column:
            return None
        with self.provider.connect() as conn:
            r = conn.select_value(f"""
                SELECT ST_Extent({conn.quote_ident(self.table.geometry_column)})
                FROM {conn.quote_table(self.table.name)}
            """)
        if not r:
            return None
        return t.Bounds(
            crs=self.table.geometry_crs,
            extent=gws.gis.extent.from_box(r))

    @property
    def props(self):
        return gws.merge(super().props, geometryType=self.table.geometry_type)

    @property
    def default_search_provider(self):
        return self.create_object('gws.ext.search.provider.postgres', t.Config(
            uid=self.uid + '.default_search',
            db=self.provider.uid,
            table=self.var('table'),
        ))

    def get_features(self, bounds, limit=0) -> t.List[t.IFeature]:
        shape = gws.gis.shape.from_bounds(bounds).transformed_to(self.table.geometry_crs)

        fs = self.provider.select(t.SelectArgs(
            table=self.table,
            shape=shape,
            limit=limit,
        ))

        return [self.connect_feature(f) for f in fs]

    def edit_operation(self, operation: str, feature_props: t.List[t.FeatureProps]) -> t.List[t.IFeature]:
        features = []

        for p in feature_props:
            if p.attributes and self.edit_data_model:
                p.attributes = self.edit_data_model.apply(p.attributes)

            features.append(gws.gis.feature.from_props(p))

        fs = self.provider.edit_operation(operation, self.table, features)
        return [self.connect_feature(f) for f in fs]

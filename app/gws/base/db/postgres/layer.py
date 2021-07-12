import gws
import gws.types as t
import gws.base.db
import gws.base.layer.vector
import gws.base.search
import gws.lib.extent
import gws.lib.feature
import gws.lib.shape

from . import provider as provider_mod


@gws.ext.Config('layer.postgres')
class Config(gws.base.layer.vector.Config):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: gws.base.db.SqlTableConfig  #: sql table configuration


@gws.ext.Object('layer.postgres')
class Object(gws.base.layer.vector.Object):
    provider: provider_mod.Object
    table: gws.SqlTable

    def configure(self):
        self.provider = provider_mod.require_provider(self)
        self.table = self.provider.configure_table(self.var('table'))
        self.is_editable = True
        if not self.data_model:
            p = self.provider.table_data_model_config(self.table)
            if p:
                self.data_model = t.cast(gws.IDataModel, self.create_child('gws.base.model', p))

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
        return gws.Bounds(
            crs=self.table.geometry_crs,
            extent=gws.lib.extent.from_box(r))

    @property
    def props(self):
        return gws.merge(super().props, geometryType=self.table.geometry_type)

    @property
    def default_search_provider(self):
        return self.root.create_object('gws.ext.search.provider.postgres', gws.Config(
            uid=self.uid + '.default_search',
            db=self.provider.uid,
            table=self.var('table'),
        ))

    def get_features(self, bounds, limit=0) -> t.List[gws.IFeature]:
        shape = gws.lib.shape.from_bounds(bounds).transformed_to(self.table.geometry_crs)

        fs = self.provider.select(gws.SqlSelectArgs(
            table=self.table,
            shape=shape,
            limit=limit,
        ))

        return [self.connect_feature(f) for f in fs]

    def edit_operation(self, operation: str, feature_props: t.List[gws.lib.feature.Props]) -> t.List[gws.IFeature]:
        features = []

        for p in feature_props:
            if p.attributes and self.edit_data_model:
                p.attributes = self.edit_data_model.apply(p.attributes)

            features.append(gws.lib.feature.from_props(p))

        fs = self.provider.edit_operation(operation, self.table, features)
        return [self.connect_feature(f) for f in fs]

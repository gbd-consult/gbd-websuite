import gws
import gws.base.db
import gws.base.layer.vector
import gws.lib.extent
import gws.lib.feature
import gws.lib.shape
import gws.types as t

from . import provider as provider_module


@gws.ext.Config('layer.postgres')
class Config(gws.base.layer.vector.Config):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: gws.base.db.SqlTableConfig  #: sql table configuration


@gws.ext.Object('layer.postgres')
class Object(gws.base.layer.vector.Object):
    provider: provider_module.Object
    table: gws.SqlTable

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

    def props_for(self, user):
        return gws.merge(super().props_for(user), geometryType=self.table.geometry_type)

    def configure_source(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.table = self.var('_table')
        else:
            self.provider = provider_module.require_for(self)
            self.table = self.provider.configure_table(self.var('table'))

        self.is_editable = True

        if not self.data_model:
            p = self.provider.table_data_model_config(self.table)
            if p:
                self.data_model = self.require_child('gws.base.model', p)

    def configure_search(self):
        if not super().configure_search():
            self.search_providers.append(
                self.root.create_object(
                    'gws.ext.search.provider.postgres',
                    gws.Config(_provider=self.provider, _table=self.table),
                    shared=True,
                    key=[self.provider.uid, self.table.name]
                ))
            return True

    def get_features(self, bounds, limit=0) -> t.List[gws.IFeature]:
        shape = gws.lib.shape.from_bounds(bounds).transformed_to(self.table.geometry_crs)

        features = self.provider.select(gws.SqlSelectArgs(
            table=self.table,
            shape=shape,
            limit=limit,
        ))

        return [f.connect_to(self) for f in features]

    def edit_operation(self, operation: str, feature_props: t.List[gws.lib.feature.Props]) -> t.List[gws.IFeature]:
        src_features = []

        for p in feature_props:
            if p.attributes and self.edit_data_model:
                p.attributes = self.edit_data_model.apply(p.attributes)
            src_features.append(gws.lib.feature.from_props(p))

        features = self.provider.edit_operation(operation, self.table, src_features)
        return [f.connect_to(self) for f in features]

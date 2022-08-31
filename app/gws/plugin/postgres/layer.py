import gws
import gws.base.database
import gws.base.layer.vector
import gws.gis.extent
import gws.base.feature
import gws.gis.crs
import gws.base.shape
import gws.types as t

from . import provider


@gws.ext.config.layer('postgres')
class Config(gws.base.layer.vector.Config):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: gws.base.database.SqlTableConfig  #: sql table configuration


@gws.ext.object.layer('postgres')
class Object(gws.base.layer.vector.Object):
    provider: provider.Object
    table: gws.SqlTable

    @property
    def own_bounds(self):
        if not self.table.geometry_column:
            return None
        with self.provider.connect() as conn:
            r = conn.select_value(
                'SELECT ST_Extent({:name}) FROM {:qname}',
                self.table.geometry_column.name,
                self.table.name)
        if not r:
            return None
        return gws.Bounds(
            crs=gws.gis.crs.get(self.table.geometry_column.srid),
            extent=gws.gis.extent.from_box(r))

    def props(self, user):
        p = super().props(user)
        if self.table.geometry_column:
            p = gws.merge(p, geometryType=self.table.geometry_column.gtype)
        return p

    def configure_source(self):
        if self.var('_provider'):
            self.provider = self.var('_provider')
            self.table = self.var('_table')
        else:
            self.provider = provider.require_for(self)
            self.table = self.provider.configure_table(self.var('table'))

        self.is_editable = True

        if not self.data_model:
            p = self.provider.table_data_model_config(self.table)
            if p:
                self.data_model = self.root.create_required('gws.base.model', p)

    def configure_search(self):
        if not super().configure_search():
            self.search_providers.append(
                self.create_child(
                    'gws.ext.search.provider.postgres',
                    gws.Config(_provider=self.provider, _table=self.table),
                    shared=True,
                    key=[self.provider.uid, self.table.name]
                ))
            return True

    def get_features(self, bounds, limit=0) -> t.List[gws.IFeature]:
        features = self.provider.select_features(gws.SqlSelectArgs(
            table=self.table,
            shape=gws.base.shape.from_bounds(bounds),
            geometry_tolerance=0,
            limit=limit,
        ))

        return [f.connect_to(self) for f in features]

    def edit_operation(self, operation: str, feature_props: t.List[gws.base.feature.Props]) -> t.List[gws.IFeature]:
        src_features = []

        for p in feature_props:
            if p.attributes and self.edit_data_model:
                p.attributes = self.edit_data_model.apply(p.attributes)
            src_features.append(gws.base.feature.from_props(p))

        features = self.provider.edit_operation(operation, self.table, src_features)
        return [f.connect_to(self) for f in features]

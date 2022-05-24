import gws.common.layer
import gws.gis.shape
import gws.gis.feature
import gws.common.db
import gws.common.model
import gws.common.search.provider
import gws.gis.extent
import gws.gis.shape
import gws.gis.feature

import gws.types as t

from . import provider


class Config(gws.common.layer.VectorConfig):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: gws.common.db.SqlTableConfig  #: sql table configuration


class Object(gws.common.layer.Vector):
    is_editable = True

    def configure(self):
        super().configure()


        self.provider = t.cast(provider.Object, gws.common.db.require_provider(self, provider.Object))
        self.db = self.provider  # for new models
        self.table = self.provider.configure_table(self.var('table'))

        # if not self.data_model:
        #     p = self.provider.table_data_model_config(self.table)
        #     if p:
        #         self.data_model = t.cast(t.IModel, self.create_child('gws.common.model', p))

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
        p = self.root.create_object('gws.ext.search.provider.postgres', t.Config(
            uid=self.uid + '.default_search',
            db=self.provider.uid,
            table=self.var('table'),
        ))
        p.model = p.model or self.model
        return p

    def get_features_ex(self, user, model, args):
        args.table = self.table
        if args.bounds:
            args.bounds = args.bounds.transformed_to(self.table.geometry_crs)

        args.table = self.table

        return model.select(args)

import gws
import gws.base.database.sql as sql
import gws.base.layer.vector
import gws.gis.extent
import gws.base.feature
import gws.gis.crs
import gws.base.shape
import gws.types as t

from . import provider, model


@gws.ext.config.layer('postgres')
class Config(gws.base.layer.Config):
    """Postgres layer"""

    db: t.Optional[str]
    """database provider uid"""
    tableName: str
    """sql table name"""


@gws.ext.object.layer('postgres')
class Object(gws.base.layer.vector.Object):
    provider: provider.Object
    tableName: str

    def configure(self):
        self.tableName = self.var('tableName')
        self.configure_provider()

        self.configure_models()

        self.configure_bounds()
        self.configure_resolutions()
        self.configure_grid()

        self.configure_legend()
        self.configure_metadata()

        self.configure_templates()
        self.configure_search()

    def configure_provider(self):
        self.provider = gws.base.database.provider.configure_for(self)
        return True

    def configure_models(self):
        if super().configure_models():
            return True
        self.models.append(self.create_child(
            gws.ext.object.model,
            gws.Config(type='postgres', _provider=self.provider, tableName=self.tableName)
        ))
        return True

    def configure_search(self):
        if super().configure_search():
            return True
        self.finders.append(self.create_child(
            gws.ext.object.finder,
            gws.Config(type='postgres', _provider=self.provider, tableName=self.tableName)
        ))
        return True

    # def configure_bounds(self):
    #     if super().configure_bounds():
    #         return True
    #     for mod in self.modelMgr.models:
    #         mod = t.cast(model.Object, model)
    #         if mod.geometryType:
    #             return self.configure_bounds_from_model(mod)
    #     return False
    #
    # def configure_bounds_from_model(self, mod):
    #     table = model.sa_t
    #     sel = sql.sa.select
    #
    #     with self.provider.session() as conn:
    #         r = conn.select_value(
    #             'SELECT ST_Extent({:name}) FROM {:qname}',
    #             self.table.geometry_column.name,
    #             self.table.name)
    #     if not r:
    #         return None
    #     return gws.Bounds(
    #         crs=gws.gis.crs.get(self.table.geometry_column.srid),
    #         extent=gws.gis.extent.from_box(r))

    def props(self, user):
        p = super().props(user)
        # if self.table.geometry_column:
        #     p = gws.merge(p, geometryType=self.table.geometry_column.gtype)
        return p

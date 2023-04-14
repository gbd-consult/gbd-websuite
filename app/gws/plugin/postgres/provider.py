"""Postgres database provider."""

import gws.base.database
import gws.gis.crs
import gws.gis.extent
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.databaseProvider('postgres')


class Config(gws.base.database.provider.Config):
    """Postgres/Postgis database provider"""


class Object(gws.base.database.provider.Object):

    def engine(self, **kwargs):
        return self.mgr.make_engine('postgresql', self.config, **kwargs)

    def qualified_table_name(self, table_name):
        if '.' in table_name:
            return table_name
        return 'public.' + table_name

    def parse_table_name(self, table_name):
        if '.' in table_name:
            schema, name = table_name.split('.')
            return schema, name
        return 'public', table_name

    def bounds_for_table(self, table_name) -> t.Optional[gws.Bounds]:
        desc = self.describe(table_name)
        if not desc.geometryName:
            return
        tab = self.table(table_name)
        with self.session() as sess:
            sel = sa.select(sa.func.ST_Extent(tab.columns.get(desc.geometryName)))
            box = sess.execute(sel).scalar_one()
            return gws.Bounds(extent=gws.gis.extent.from_box(box), crs=gws.gis.crs.get(desc.geometrySrid))

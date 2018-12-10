import gws
import gws.config
import gws.tools.json2
import gws.gis.proj
import gws.gis.feature
import gws.gis.shape
import gws.types as t


class Config(t.Config):
    """SQL source"""

    #: object type
    type: str
    #: db provider id
    db: str
    #: sql table configuration
    table: t.SqlTableConfig


class Object(gws.Object):
    provider: t.DbProviderObject = None

    def configure(self):
        super().configure()

        self.provider = self.root.find('gws.ext.db.provider', self.var('db'))
        if not self.provider:
            raise gws.Error(f'{self.uid}: db provider not found')

    def get_features(self, keyword, shape: gws.gis.shape.Shape, sort=None, limit=None):
        return self.provider.select(t.SelectArgs({
            'table': self.var('table'),
            'keyword': keyword,
            'shape': shape,
            'sort': sort,
            'limit': limit
        }))

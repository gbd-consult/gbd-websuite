import gws.common.layer
import gws.gis.shape
import gws.gis.feature
import gws.common.db
import gws.tools.misc as misc

import gws.types as t

from . import provider, util


class Config(gws.common.layer.VectorConfig):
    """SQL-based layer"""

    db: t.Optional[str]  #: database provider uid
    table: t.SqlTableConfig  #: sql table configuration


class Object(gws.common.layer.Vector):
    def __init__(self):
        super().__init__()
        self.provider: provider.Object = None
        self.table: t.SqlTable = None

    def configure(self):
        super().configure()

        self.provider: provider.Object = gws.common.db.require_provider(self, provider.Object)
        self.table = util.configure_table(self, self.provider)

        p = self.var('search')
        if not p or (p.enabled and not p.providers):
            # spatial search by default
            self.add_child('gws.ext.search.provider.postgres', t.Config({
                'db': self.provider.uid,
                'table': self.var('table'),
                'geometryRequired': True,
            }))

    @property
    def props(self):
        return super().props.extend({
            'geometryType': self.table.geometry_type.upper(),
        })

    def get_features(self, bbox, limit=0) -> t.List[t.Feature]:
        shape = gws.gis.shape.from_bbox(bbox, self.crs)

        fs = self.provider.select(t.SelectArgs({
            'table': self.table,
            'shape': shape,
            'limit': limit,
        }))

        return [self.connect_feature(f) for f in fs]

    def edit_operation(self, operation: str, feature_props: t.List[t.FeatureProps]) -> t.List[t.Feature]:
        features = []

        for p in feature_props:
            if p.attributes and self.edit_data_model:
                p.attributes = self.edit_data_model.apply(p.attributes)

            features.append(gws.gis.feature.from_props(p))

        fs = self.provider.edit_operation(operation, self.table, features)
        return [self.connect_feature(f) for f in fs]

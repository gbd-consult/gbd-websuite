"""Database-based models."""

import gws
import gws.base.feature
import gws.base.model
import gws.base.model.field
import gws.gis.crs
import gws.lib.sa as sa

import gws.types as t

from . import provider


class Props(gws.base.model.Props):
    pass


class Config(gws.base.model.Config):
    dbUid: t.Optional[str]
    """db provider uid"""
    tableName: t.Optional[str]
    """table name for the model"""
    sqlFilter: t.Optional[str]
    """extra SQL filter"""


class Object(gws.base.model.Object, gws.IDatabaseModel):
    provider: gws.IDatabaseProvider

    def configure(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        if not self.tableName:
            raise gws.ConfigurationError(f'table name missing in model {self!r}')

        self.sqlFilter = self.cfg('sqlFilter')
        self.configure_model()

    def configure_provider(self):
        self.provider = t.cast(gws.IDatabaseProvider, provider.get_for(self, ext_type=self.extType))
        return True

    ##

    def describe(self):
        return self.provider.describe(self.tableName)

    def table(self):
        return self.provider.table(self.tableName)

    def column(self, column_name):
        return self.provider.column(self.table(), column_name)

    def uid_column(self):
        if not self.uidName:
            raise gws.Error(f'no primary key found for table {self.tableName!r}')
        if not self.provider.has_column(self.table(), self.uidName):
            raise gws.Error(f'invalid primary key {self.uidName!r} for table {self.tableName!r}')
        return self.provider.column(self.table(), self.uidName)

    def connection(self):
        return self.provider.connection()

    def _context_connection(self, mc):
        return _ContextConnection(self, mc)

    def execute(self, sql, mc, parameters=None) -> sa.CursorResult:
        with self._context_connection(mc):
            return mc.dbConnection.execute(sql, parameters or [])

    def commit(self, mc) -> sa.CursorResult:
        with self._context_connection(mc):
            return mc.dbConnection.commit()

    ##

    def find_features(self, search, mc):
        if not mc.user.can_read(self):
            raise gws.ForbiddenError()

        mc.search = search
        mc.dbSelect = gws.ModelDbSelect(
            columns=[],
            geometryWhere=[],
            keywordWhere=[],
            order=[],
            where=[]
        )

        with self._context_connection(mc):
            for fld in self.fields:
                fld.before_select(mc)

            sql = self._make_select(mc)
            if sql is None:
                gws.log.debug('empty select')
                return []

            features = []

            for row in self.execute(sql, mc):
                features.append(gws.base.feature.new(
                    model=self,
                    record=gws.FeatureRecord(attributes=row._asdict())
                ))

            for fld in self.fields:
                fld.after_select(features, mc)

        return features

    def _make_select(self, mc: gws.ModelContext) -> t.Optional[sa.Select]:

        # @TODO this should happen on the field level
        sorts = mc.search.sort or self.defaultSort or []
        for s in sorts:
            fn = sa.desc if s.reverse else sa.asc
            mc.dbSelect.order.append(fn(self.column(s.fieldName)))

        sel = sa.select().select_from(self.table())

        if mc.search.uids:
            if not self.uidName:
                return
            sel = sel.where(self.uid_column().in_(mc.search.uids))

        if mc.search.keyword and not mc.dbSelect.keywordWhere:
            return
        if mc.dbSelect.keywordWhere:
            sel = sel.where(sa.or_(*mc.dbSelect.keywordWhere))

        if mc.search.shape and not mc.dbSelect.geometryWhere:
            return
        if mc.dbSelect.geometryWhere:
            sel = sel.where(sa.or_(*mc.dbSelect.geometryWhere))

        sel = sel.where(*mc.dbSelect.where)
        if mc.search.extraWhere:
            for w in mc.search.extraWhere:
                sel = sel.where(w)

        if self.sqlFilter:
            sel = sel.where(sa.text('(' + self.sqlFilter + ')'))

        cols = []
        for col in mc.dbSelect.columns or []:
            if any(col is c for c in cols):
                continue
            cols.append(col)
        for col in mc.search.extraColumns or []:
            if any(col is c for c in cols):
                continue
            cols.append(col)

        sel = sel.add_columns(*cols)

        if mc.dbSelect.order:
            sel = sel.order_by(*mc.dbSelect.order)

        return sel

    ##

    def init_feature(self, feature, mc):
        if not mc.user.can_create(self):
            raise gws.ForbiddenError()

        for fld in self.fields:
            fld.do_init(feature, mc)

        for rf in feature.createWithFeatures:
            for fld in rf.model.fields:
                fld.do_init_related(feature, mc)

    def create_feature(self, feature, mc):
        if not mc.user.can_create(self):
            raise gws.ForbiddenError()

        feature.record = gws.FeatureRecord(attributes={}, meta={})

        related_models = []
        for from_feature in feature.createWithFeatures:
            if from_feature.model not in related_models:
                related_models.append(from_feature.model)

        with self._context_connection(mc):
            for m in related_models:
                for fld in m.fields:
                    fld.before_create_related(feature, mc)

            for fld in self.fields:
                fld.before_create(feature, mc)

            sql = sa.insert(self.table())
            rs = self.execute(sql, mc, feature.record.attributes)
            feature.insertedPrimaryKey = rs.inserted_primary_key[0]

            for fld in self.fields:
                fld.after_create(feature, mc)

            for m in related_models:
                for fld in m.fields:
                    fld.after_create_related(feature, mc)

            self.commit(mc)

        return feature.insertedPrimaryKey

    def update_feature(self, feature, mc):
        if not mc.user.can_write(self):
            raise gws.ForbiddenError()

        feature.record = gws.FeatureRecord(attributes={}, meta={})

        with self._context_connection(mc):
            for fld in self.fields:
                fld.before_update(feature, mc)

            if not feature.record.attributes:
                return feature.uid()

            sql = self.table().update().where(
                self.uid_column().__eq__(feature.uid())
            ).values(
                feature.record.attributes
            )
            self.execute(sql, mc)

            for fld in self.fields:
                fld.after_update(feature, mc)

            self.commit(mc)

        return feature.uid()

    def delete_feature(self, feature, mc):
        if not mc.user.can_delete(self):
            raise gws.ForbiddenError()

        with self._context_connection(mc):
            for fld in self.fields:
                fld.before_delete(feature, mc)

            sql = sa.delete(self.table()).where(
                self.uid_column().__eq__(feature.uid())
            )

            self.execute(sql, mc)

            for fld in self.fields:
                fld.after_delete(feature, mc)

            self.commit(mc)

        return feature.uid()


class _ContextConnection:
    def __init__(self, model: Object, mc: gws.ModelContext):
        self.mc = mc
        self.isTop = self.mc.dbConnection is None
        if self.isTop:
            self.mc.dbConnection = model.connection()

    def __enter__(self) -> sa.Connection:
        return self.mc.dbConnection

    def __exit__(self, typ, value, traceback):
        if self.isTop:
            self.mc.dbConnection.close()
            self.mc.dbConnection = None

"""Base database model."""

from typing import Optional, Iterable

import gws
import gws.base.feature
import gws.base.model
import gws.config.util
import gws.lib.sa as sa


class Config(gws.base.model.Config):
    """Configuration for the database model."""

    dbUid: Optional[str]
    """UID of the database provider."""
    tableName: Optional[str]
    """Database table of the model."""
    sqlFilter: Optional[str]
    """SQL condition added to every query of the model."""


class Props(gws.base.model.Props):
    pass


class Object(gws.base.model.Object, gws.DatabaseModel):
    """Base model for a database table.

    Reads features from one table with SELECT queries built from the model
    fields, and creates, updates and deletes table rows. Provides subclasses
    with the database provider, access to the table and its columns, and the
    SELECT builder.
    """

    def configure(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        if not self.tableName:
            raise gws.ConfigurationError(f'table name missing in model {self!r}')

        self.sqlFilter = self.cfg('sqlFilter')
        self.configure_model()

    def configure_provider(self):
        return gws.config.util.configure_database_provider_for(self)

    ##

    def describe(self):
        return self.db.describe(self.tableName)

    def table(self):
        return self.db.table(self.tableName)

    def column(self, column_name):
        return self.db.column(self.table(), column_name)

    def uid_column(self):
        if not self.uidName:
            raise gws.Error(f'no primary key found for table {self.tableName!r}')
        if not self.db.has_column(self.table(), self.uidName):
            raise gws.Error(f'invalid primary key {self.uidName!r} for table {self.tableName!r}')
        return self.db.column(self.table(), self.uidName)

    def uid_equals(self, uid):
        col = self.uid_column()
        if isinstance(uid, (str, bytes)) or not isinstance(uid, Iterable):
            return col == sa.bindparam(None, uid, type_=col.type)
        return col.in_(sa.bindparam(None, list(uid), expanding=True, type_=col.type))

    ##

    def find_features(self, search, mc):
        if not mc.user.can_read(self):
            raise gws.ForbiddenError(f'model {self.uid!r} can_read=False')
        
        mc = gws.base.model.copy_context(mc)
        mc.search = search
        mc.dbSelect = gws.ModelSelectBuild(
            columns=[],
            geometryWhere=[],
            keywordWhere=[],
            order=[],
            where=[],
        )

        with self.db.begin():
            for fld in self.fields:
                fld.before_select(mc)

            sql = self.build_select(mc)
            if sql is None:
                return []

            features = self.fetch_features(sql)

            for fld in self.fields:
                fld.after_select(features, mc)

        return features

    def fetch_features(self, select):
        features = []

        with self.db.begin() as conn:
            for row in conn.fetch_all(select):
                features.append(
                    gws.base.feature.new(
                        model=self,
                        record=gws.FeatureRecord(attributes=row),
                    )
                )

        return features

    def build_select(self, mc):
        # @TODO sorting should be handled on the field level
        sorts = mc.search.sort or self.defaultSort or []
        for s in sorts:
            fn = sa.desc if s.reverse else sa.asc
            mc.dbSelect.order.append(fn(self.column(s.fieldName)))

        sel = sa.select().select_from(self.table())

        if mc.search.uids:
            if not self.uidName:
                gws.log.debug(f'build_select: {self}: no primary key for {self.tableName=}')
                return
            sel = sel.where(self.uid_equals(mc.search.uids))

        if mc.search.keyword and not mc.dbSelect.keywordWhere:
            gws.log.debug(f'build_select: {self}: no keyword where')
            return
        if mc.dbSelect.keywordWhere:
            sel = sel.where(sa.or_(*mc.dbSelect.keywordWhere))

        if mc.search.shape and not mc.dbSelect.geometryWhere:
            gws.log.debug(f'build_select: {self}: no geometry where')
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

        if mc.search.limit:
            sel = sel.limit(mc.search.limit)

        return sel

    ##

    def init_feature(self, feature, mc):
        if not mc.user.can_create(self):
            raise gws.ForbiddenError(f'model {self.uid!r} can_create=False')

        for fld in self.fields:
            fld.do_init(feature, mc)

        for rf in feature.createWithFeatures:
            for fld in rf.model.fields:
                fld.do_init_related(feature, mc)

        feature.isNew = True

    def create_feature(self, feature, mc):
        if not mc.user.can_create(self):
            raise gws.ForbiddenError(f'model {self.uid!r} can_create=False')

        feature.record = gws.FeatureRecord(attributes={}, meta={})

        related_models = []
        for from_feature in feature.createWithFeatures:
            if from_feature.model not in related_models:
                related_models.append(from_feature.model)

        with self.db.begin() as conn:
            for m in related_models:
                for fld in m.fields:
                    fld.before_create_related(feature, mc)

            for fld in self.fields:
                fld.before_create(feature, mc)

            sql = sa.insert(self.table())
            rs = conn.execute(sql, feature.record.attributes)
            pk = rs.inserted_primary_key
            if not pk:
                feature.insertedPrimaryKey = None
            elif len(pk) == 1:
                feature.insertedPrimaryKey = pk[0]
            else:
                raise gws.Error(f'composite primary keys not supported for {self.tableName!r}')

            for fld in self.fields:
                fld.after_create(feature, mc)

            for m in related_models:
                for fld in m.fields:
                    fld.after_create_related(feature, mc)

        return feature.insertedPrimaryKey

    def update_feature(self, feature, mc):
        if not mc.user.can_write(self):
            raise gws.ForbiddenError(f'model {self.uid!r} can_write=False')

        feature.record = gws.FeatureRecord(attributes={}, meta={})

        with self.db.begin() as conn:
            for fld in self.fields:
                fld.before_update(feature, mc)

            if not feature.record.attributes:
                return feature.uid()

            sql = self.table().update().where(self.uid_equals(feature.uid())).values(feature.record.attributes)
            conn.execute(sql)

            for fld in self.fields:
                fld.after_update(feature, mc)

        return feature.uid()

    def delete_feature(self, feature, mc):
        if not mc.user.can_delete(self):
            raise gws.ForbiddenError(f'model {self.uid!r} can_delete=False')

        with self.db.begin() as conn:
            for fld in self.fields:
                fld.before_delete(feature, mc)

            sql = sa.delete(self.table()).where(self.uid_equals(feature.uid()))

            conn.execute(sql)

            for fld in self.fields:
                fld.after_delete(feature, mc)

        return feature.uid()

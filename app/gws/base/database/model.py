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


class SortConfig:
    fieldName: str
    reverse: bool = False


class Config(gws.base.model.Config):
    dbUid: t.Optional[str]
    """db provider uid"""
    tableName: t.Optional[str]
    """table name for the model"""
    filter: t.Optional[str]
    """extra SQL filter"""
    sort: t.Optional[list[SortConfig]]
    """default sorting"""


class Object(gws.base.model.Object, gws.IDatabaseModel):
    provider: provider.Object

    def configure(self):
        self.sqlFilter = self.cfg('filter')
        self.sqlSort = [gws.SearchSort(c) for c in self.cfg('sort', default=[])]
        self.configure_provider()
        self.configure_sources()
        self.configure_fields()
        self.configure_key()
        self.configure_geometry()
        self.configure_templates()

    def configure_provider(self):
        self.provider = t.cast(provider.Object, provider.get_for(self, ext_type=self.extType))
        self.provider.mgr.register_model(self)
        return True

    def configure_sources(self):
        self.tableName = self.cfg('tableName') or self.cfg('_defaultTableName')
        if not self.tableName:
            raise gws.Error(f'table not found for model {self.uid!r}')

    ##

    def describe(self):
        with self.provider.session() as sess:
            return sess.describe(self.tableName)

    def table(self):
        return self.provider.mgr.table_for_model(self)

    def record_class(self):
        return self.provider.mgr.class_for_model(self)

    def get_record(self, uid):
        with self.session() as sess:
            return sess.saSession.get(self.record_class(), uid)

    def get_records(self, uids):
        with self.session() as sess:
            cls = self.record_class()
            sel = sa.select(cls).where(self.primary_keys()[0].in_(uids))
            res = sess.execute(sel)
            return res.unique().scalars().all()

    def primary_keys(self):
        return self.provider.mgr.primary_keys_for_model(self)

    def find_features(self, search, user, **kwargs):

        sel = self.make_select(search, user)
        if not sel:
            gws.log.debug('empty select')
            return []

        features = []

        with self.session() as sess:
            res = sess.execute(sel.saSelect)
            for record in res.unique().scalars().all():
                features.append(
                    self.feature_from_record(record, user, search.relationDepth or 0, **kwargs))

        return features

    def write_feature(self, feature, user, **kwargs):
        access = gws.Access.create if feature.isNew else gws.Access.write

        if not user.can(access, self):
            raise gws.ForbiddenError('forbidden')

        feature.errors = []

        for f in self.fields:
            if f.compute(feature, access, user, **kwargs):
                continue
            if user.can(access, f):
                f.validate(feature, access, user, **kwargs)

        if len(feature.errors) > 0:
            return False

        cls = self.record_class()

        with self.session() as sess:
            record = cls() if feature.isNew else sess.saSession.get(cls, feature.uid())
            for f in self.fields:
                if user.can(access, f, self):
                    f.store_to_record(feature, record, user)
            if feature.isNew:
                sess.saSession.add(record)
            sess.commit()

        return True

    def delete_feature(self, feature, user, **kwargs):
        if not user.can_delete(self):
            raise ValueError('Forbidden')

        cls = self.record_class()

        with self.session() as sess:
            record = sess.saSession.get(cls, feature.uid())
            sess.saSession.delete(record)
            sess.commit()

        return True

    ##

    def session(self) -> gws.IDatabaseSession:
        return self.provider.session()

    def make_select(self, search: gws.SearchQuery, user: gws.IUser):
        cls = self.record_class()

        sel = gws.SelectStatement(
            saSelect=sa.select(cls),
            search=search,
            keywordWhere=[],
            geometryWhere=[],
        )

        for f in self.fields:
            f.augment_select(sel, user)

        if search.keyword and not sel.keywordWhere:
            return
        if sel.keywordWhere:
            sel.saSelect = sel.saSelect.where(sa.or_(*sel.keywordWhere))

        if search.shape and not sel.geometryWhere:
            return
        if sel.geometryWhere:
            sel.saSelect = sel.saSelect.where(sa.or_(*sel.geometryWhere))

        if search.uids:
            pks = self.primary_keys()
            if not pks:
                return
            pk = getattr(cls, pks[0].name)
            sel.saSelect = sel.saSelect.where(pk.in_(search.uids))

        if search.extraWhere:
            for c in search.extraWhere:
                where = sa.text(c.text)
                if c.args:
                    where = where.bindparams(**c.args)
                sel.saSelect = sel.saSelect.where(where)

        if self.sqlFilter:
            sel.saSelect = sel.saSelect.where(sa.text(self.sqlFilter))

        sorter = search.sort or self.sqlSort
        if sorter:
            order = []
            for s in sorter:
                fn = sa.desc if s.reverse else sa.asc
                order.append(fn(getattr(cls, s.fieldName)))
            sel.saSelect = sel.saSelect.order_by(*order)

        return sel

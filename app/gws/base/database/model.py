"""Database-based models."""

import sqlalchemy as sa

import gws.base.feature
import gws.base.model
import gws.base.model.field
import gws.gis.crs

import gws.types as t

from . import provider


class Props(gws.base.model.Props):
    pass


class Config(gws.base.model.Config):
    db: t.Optional[str]
    tableName: t.Optional[str]


class Object(gws.base.model.Object):
    provider: provider.Object
    tableName: str

    def configure(self):
        self.tableName = self.cfg('tableName')

        self.configure_provider()
        self.configure_fields()
        self.configure_properties()

    def configure_provider(self):
        self.provider = t.cast(provider.Object, provider.get_for(self, ext_type=self.extType))
        self.provider.mgr.register_model(self)

    def configure_fields(self):
        if super().configure_fields():
            return True
        with self.provider.session() as sess:
            desc = sess.describe(self.tableName)
        for column in desc.columns.values():
            cfg = gws.base.model.field.config_from_column(column)
            self.fields.append(
                self.create_child(gws.ext.object.modelField, config=gws.merge(cfg, _model=self)))
        return True

    def configure_properties(self):
        self.geometryType = None
        self.geometryCrs = None

        if not self.keyName:
            for f in self.fields:
                if f.isPrimaryKey:
                    self.keyName = f.name
                    break

        geom = None

        for f in self.fields:
            if self.geometryName and f.name == self.geometryName:
                geom = f
                break
            if not self.geometryName and f.attributeType == gws.AttributeType.geometry:
                geom = f
                break

        if geom:
            self.geometryName = geom.name
            if not geom.geometryType:
                with self.provider.session() as sess:
                    desc = sess.describe(self.tableName)
                geom.geometryType = desc.columns[geom.name].geometryType
                geom.geometryCrs = gws.gis.crs.get(desc.columns[geom.name].geometrySrid)
            self.geometryType = geom.geometryType
            self.geometryCrs = geom.geometryCrs

    def table(self) -> sa.Table:
        return self.provider.mgr.table_for_model(self)

    def orm_class(self):
        return self.provider.mgr.class_for_model(self)

    def primary_keys(self):
        return self.provider.mgr.pkeys_for_model(self)

    def find_features(self, search, user, **kwargs):

        sel = self.build_select(search, user)
        if not sel:
            gws.log.debug('empty select')
            return []

        features = []

        with self.session() as sess:
            cursor = sess.execute(sel.saSelect)
            for row in cursor.unique().all():
                record = getattr(row, self.orm_class().__name__)
                features.append(self.feature_from_record(record, user, **kwargs))

        return features

    def write_features(self, features, user, **kwargs):
        for feature in features:
            access = gws.Access.create if feature.isNew else gws.Access.write

            if not user.can(access, self):
                raise ValueError('Forbidden')

            feature.errors = []

            for f in self.fields:
                if f.compute(feature, access, user, **kwargs):
                    continue
                if user.can(access, f):
                    f.validate(feature, access, user, **kwargs)

        if any(len(f.errors) > 0 for f in features):
            return False

        rmap = {}
        cls = self.orm_class()

        with self.session() as sess:
            with sess.begin():
                for feature in features:
                    record = cls() if feature.isNew else sess.sa.get(cls, feature.uid())
                    access = gws.Access.create if feature.isNew else gws.Access.write
                    for f in self.fields:
                        if user.can(access, f, self):
                            f.store_to_record(feature, record, user)
                    if feature.isNew:
                        sess.sa.add(record)
                    rmap[id(feature)] = record

                sess.commit()

            for feature in features:
                record = rmap[id(feature)]
                feature.attributes = {}
                for f in self.fields:
                    f.load_from_record(feature, record, user)

        return True

    def delete_features(self, features, user, **kwargs):
        if not user.can_delete(self):
            raise ValueError('Forbidden')

        cls = self.orm_class()

        with self.session() as sess:
            with sess.begin():
                for feature in features:
                    record = sess.sa.get(cls, feature.uid())
                    sess.sa.delete(record)
                sess.commit()

        return True

    ##

    def session(self) -> gws.IDatabaseSession:
        return self.provider.session()

    def build_select(self, search: gws.SearchArgs, user: gws.IUser):
        sel = gws.SelectStatement(
            saSelect=sa.select(self.orm_class()),
            search=search,
            keywordWhere=[],
            geometryWhere=[],
        )

        for f in self.fields:
            f.select(sel, user)

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
            pk = getattr(self.orm_class(), pks[0].name)
            sel.saSelect = sel.saSelect.where(pk.in_(search.uids))

        # extra = (self.extraWhere or [])
        #
        # if args.extra_where:
        #     s = sa.text(args.extra_where[0])
        #     if len(args.extra_where) > 1:
        #         s = s.bindparams(**args.extra_where[1])
        #     select.saSelect = select.saSelect.where(s)
        #
        # if self.filter:
        #     select.saSelect = select.saSelect.where(sa.text(self.filter))

        if search.sort:
            for s in search.sort:
                fn = sa.desc if s.reverse else sa.asc
                sel.saSelect = sel.saSelect.order_by(fn(getattr(self.orm_class(), s.fieldName)))

        gws.log.debug(f'make_select: {str(sel.saSelect)}')

        return sel

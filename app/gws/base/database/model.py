"""Database-based models."""

import gws.base.feature
import gws.base.model
import gws.base.model.field
import gws.gis.crs

import gws.types as t

from . import provider, sql


class Props(gws.base.model.Props):
    pass


class Config(gws.base.model.Config):
    db: t.Optional[str]
    tableName: str


class Object(gws.base.model.Object):
    provider: provider.Object
    tableName: str

    def configure(self):
        self.tableName = self.var('tableName')

        self.configure_provider()
        self.configure_fields()
        self.configure_properties()

    def configure_provider(self):
        self.provider = provider.get_for(self, ext_type=self.extType)
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

    def sa_table(self):
        return self.provider.mgr.table_for_model(self)

    def sa_class(self):
        return self.provider.mgr.class_for_model(self)

    saTable: sql.sa.Table
    saPrimaryKeyColumns: t.List[sql.sa.Column]

    def find_features(self, search, user, **kwargs):

        sel = self.build_select(search, user)
        if not sel:
            gws.log.debug('empty select')
            return []

        features = []
        session = self.session()

        cursor = session.saSession.execute(sel.saSelect)
        for row in cursor.unique().all():
            record = getattr(row, self.sa_class().__name__)
            features.append(self.feature_from_record(record, user, **kwargs))

        return features

    def write_features(self, features, user, **kwargs):
        error_cnt = 0

        for feature in features:
            access = gws.Access.create if feature.isNew else gws.Access.write

            if not user.can(access, self):
                error_cnt += 1
                continue

            feature.errors = []

            for f in self.fields:
                if f.compute(feature, access, user, **kwargs):
                    continue
                if user.can(access, f):
                    f.validate(feature, access, user, **kwargs)

            error_cnt += len(feature.errors)

        if error_cnt:
            return False

        records = {}
        session = self.session()

        for feature in features:
            cls = self.sa_class()
            record = cls() if feature.isNew else session.saSession.get(cls, feature.uid())
            access = gws.Access.create if feature.isNew else gws.Access.write
            for f in self.fields:
                if user.can(access, f, self):
                    f.store_to_record(feature, record, user)
            if feature.isNew:
                session.saSession.add(record)
            records[id(feature)] = record

        session.commit()

        for feature in features:
            record = records[id(feature)]
            feature.attributes = {}
            for f in self.fields:
                f.load_from_record(feature, record, user)

        return True

    ##

    def session(self) -> sql.Session:
        return t.cast(sql.Session, self.provider.session())

    def build_select(self, search: gws.SearchArgs, user: gws.IUser):
        sel = sql.SelectStatement(
            saSelect=sql.sa.select(self.sa_class()),
            search=search,
            keywordWhere=[],
            geometryWhere=[],
        )

        for f in self.fields:
            f.sa_select(sel, user)

        if search.keyword and not sel.keywordWhere:
            return
        if sel.keywordWhere:
            sel.saSelect = sel.saSelect.where(sql.sa.or_(*sel.keywordWhere))

        if search.shape and not sel.geometryWhere:
            return
        if sel.geometryWhere:
            sel.saSelect = sel.saSelect.where(sql.sa.or_(*sel.geometryWhere))

        if search.uids:
            pk = getattr(self.sa_class(), self.saPrimaryKeyColumns[0].name)
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
                fn = sql.sa.desc if s.reverse else sql.sa.asc
                sel.saSelect = sel.saSelect.order_by(fn(getattr(self.sa_class(), s.fieldName)))

        gws.log.debug(f'make_select: {str(sel.saSelect)}')

        return sel

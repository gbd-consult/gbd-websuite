"""Database-based models."""

import gws.base.feature
import gws.base.model
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
        self.provider = provider.configure_for(self, ext_type=self.extType)
        self.provider.mgr.register_model(self)

    def configure_fields(self):
        if super().configure_fields():
            return True
        desc = self.provider.describe_table(self.tableName)

    def configure_properties(self):
        self.geometryType = None
        self.geometryCrs = None

        for f in self.fields:
            if f.isPrimaryKey and not self.keyName:
                self.keyName = f.name
            if f.attributeType == gws.AttributeType.geometry and not self.geometryName:
                self.geometryName = f.name
                self.geometryType = f.geometryType
                self.geometryCrs = f.geometryCrs

    def sa_table(self):
        return self.provider.mgr.sa_table(self)

    def sa_class(self):
        return self.provider.mgr.sa_class(self)

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
            feature = gws.base.feature.with_model(self)
            record = getattr(row, self.sa_class().__name__)
            for f in self.fields:
                f.load_from_record(feature, record, user)
            features.append(feature)

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
                if f.compute_value(feature, access, user, **kwargs):
                    continue
                if user.can(access, f):
                    f.validate_value(feature, access, user, **kwargs)

            error_cnt += len(feature.errors)

        if error_cnt:
            return False

        records = {}
        session = self.session()

        for feature in features:
            cls = self.sa_class()
            record = cls() if feature.isNew else session.saSession.get(cls, self.keyof(feature))
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

    def keyof(self, feature):
        return feature.attributes.get(self.keyName)

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
        # for s in self.sort:
        #     fn = sa.desc if s.order == 'desc' else sa.asc
        #     state.sel = state.sel.order_by(fn(getattr(cls, s.fieldName)))

        gws.log.debug(f'make_select: {str(sel.saSelect)}')

        return sel

"""Generic related field."""

import gws
import gws.base.model.util as mu
import gws.base.model.field
import gws.lib.sa as sa
import gws.types as t


class Config(gws.base.model.field.Config):
    pass


class Props(gws.base.model.field.Props):
    pass


class Link(gws.Data):
    table: sa.Table
    fromKey: sa.Column
    toKey: sa.Column


class RelRef(gws.Data):
    model: gws.DatabaseModel
    table: sa.Table
    key: sa.Column
    uid: sa.Column


class Relationship(gws.Data):
    src: RelRef
    to: RelRef
    tos: list[RelRef]
    link: Link
    deleteCascade: bool = False


class Object(gws.base.model.field.Object, gws.ModelField):
    model: gws.DatabaseModel
    rel: Relationship

    def __getstate__(self):
        return gws.u.omit(vars(self), 'rel')

    def post_configure(self):
        self.configure_relationship()

    def activate(self):
        self.configure_relationship()

    def configure_relationship(self):
        pass

    def configure_widget(self):
        if not super().configure_widget():
            if self.attributeType == gws.AttributeType.feature:
                self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='featureSelect')
                return True
            if self.attributeType == gws.AttributeType.featurelist:
                self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='featureList')
                return True

    def get_model(self, uid: str) -> gws.DatabaseModel:
        mod = self.root.get(uid)
        if not mod:
            raise gws.ConfigurationError(f'model {uid!r} not found')
        return t.cast(gws.DatabaseModel, mod)

    def find_relatable_features(self, search, mc):
        return [
            f
            for to in self.rel.tos
            for f in to.model.find_features(search, mc)
        ]

    def related_field(self, to: RelRef) -> t.Optional[gws.ModelField]:
        for fld in to.model.fields:
            rel2 = t.cast(Relationship, getattr(fld, 'rel', None))
            if not rel2:
                continue
            if rel2.src.model == to.model and rel2.src.key == to.key:
                return fld

    def do_init_related(self, to_feature, mc):
        our_features = [f for f in to_feature.createWithFeatures if f.model == self.model]
        if not our_features:
            return

        for to in self.rel.tos:
            if to.model == to_feature.model:
                fld = self.related_field(to)
                if fld:
                    if fld.attributeType == gws.AttributeType.feature:
                        to_feature.attributes[fld.name] = our_features[0]
                    if fld.attributeType == gws.AttributeType.featurelist:
                        to_feature.attributes.setdefault(fld.name, []).extend(our_features)

    def related_models(self):
        return [to.model for to in self.rel.tos]

    ##

    def to_props(self, feature, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        value = feature.get(self.name)
        if not value:
            return
        if not isinstance(value, list):
            value = [value]

        mc2 = mu.secondary_context(mc)
        res = []

        for v in value:
            related = t.cast(gws.Feature, v)
            if related:
                p = related.model.feature_to_props(related, mc2)
                if p:
                    res.append(p)

        if self.attributeType == gws.AttributeType.featurelist:
            feature.props.attributes[self.name] = res
        elif res:
            feature.props.attributes[self.name] = res[0]

    def from_props(self, feature, mc):
        if mc.relDepth >= mc.maxDepth:
            return

        value = feature.props.attributes.get(self.name)
        if not value:
            return
        if not isinstance(value, list):
            value = [value]

        mc2 = mu.secondary_context(mc)
        res = []
        to_model_map: dict[str, gws.Model] = {to.model.uid: to.model for to in self.rel.tos}

        for v in value:
            rel_props = t.cast(gws.FeatureProps, gws.u.to_data_object(v))
            if rel_props:
                to_model = to_model_map.get(rel_props.modelUid)
                if to_model:
                    related = to_model.feature_from_props(rel_props, mc2)
                    if related:
                        res.append(related)

        if self.attributeType == gws.AttributeType.featurelist:
            feature.set(self.name, res)
        elif res:
            feature.set(self.name, res[0])

    ##

    def key_for_uid(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uid: gws.FeatureUid,
            mc: gws.ModelContext
    ):
        sql = sa.select(key_column).where(model.uid_column().__eq__(uid))
        rs = list(self.model.execute(sql, mc))
        return rs[0][0] if rs else None

    def update_key_for_uids(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: list[gws.FeatureUid],
            key: t.Any,
            mc: gws.ModelContext
    ):

        sql = sa.update(
            model.table()
        ).values({
            key_column: key
        }).where(
            model.uid_column().in_(uids)
        )
        self.model.execute(sql, mc)

    def uids_to_keys(
            self,
            mc: gws.ModelContext,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: t.Optional[t.Iterable[gws.FeatureUid]] = None,
            keys: t.Optional[t.Iterable[t.Any]] = None,
    ):

        if uids:
            uids = set(v for v in uids if v is not None)
            sql = sa.select(model.uid_column(), key_column).where(model.uid_column().in_(uids))
        else:
            keys = set(v for v in keys if v is not None)
            sql = sa.select(model.uid_column(), key_column).where(key_column.in_(keys))

        return {str(uid): key for uid, key in self.model.execute(sql, mc)}

    def uid_and_key_for_uids(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: t.Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> set[tuple[gws.FeatureUid, gws.FeatureUid]]:

        vs = set(v for v in uids if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(model.uid_column().in_(vs))
        return set((uid, key) for uid, key in self.model.execute(sql, mc))

    def uid_and_key_for_keys(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            keys: t.Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> set[tuple[gws.FeatureUid, gws.FeatureUid]]:

        vs = set(v for v in keys if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(key_column.in_(vs))
        return set((uid, key) for uid, key in self.model.execute(sql, mc))

    def update_uid_and_key(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uid_and_key: t.Iterable[tuple[gws.FeatureUid, gws.FeatureUid]],
            mc: gws.ModelContext
    ):

        for uid, key in uid_and_key:
            sql = sa.update(
                model.table()
            ).values({
                key_column: key
            }).where(
                model.uid_column().__eq__(uid)
            )
            self.model.execute(sql, mc)

    def drop_uid_and_key(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: t.Iterable[gws.FeatureUid],
            delete: bool,
            mc: gws.ModelContext,
    ):

        vs = set(v for v in uids if v is not None)
        if not vs:
            return

        if delete:
            sql = sa.delete(
                model.table()
            ).where(
                model.uid_column().in_(vs)
            )
        else:
            sql = sa.update(
                model.table()
            ).values({
                key_column.name: None
            }).where(
                model.uid_column().in_(vs)
            )

        self.model.execute(sql, mc)

    def get_related(
            self,
            model: gws.DatabaseModel,
            uids: t.Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> list[gws.Feature]:
        return model.get_features(uids, mu.secondary_context(mc))

    def column_or_uid(self, model, cfg):
        return model.column(cfg) if cfg else model.uid_column()

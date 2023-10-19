"""Generic related field."""

import gws
import gws.base.model.util as mu
import gws.lib.sa as sa
import gws.types as t
from . import field


class Config(field.Config):
    pass


class Props(field.Props):
    pass


class Link(gws.Data):
    table: sa.Table
    fromColumn: sa.Column
    toColumn: sa.Column


class RelRef(gws.Data):
    model: gws.IDatabaseModel
    table: sa.Table
    column: sa.Column
    uid: sa.Column


class Relationship(gws.Data):
    src: RelRef
    to: list[RelRef]
    link: Link
    deleteCascade: bool = False


class Object(field.Object, gws.IModelRelatedField):
    model: gws.IDatabaseModel
    rel: Relationship

    def __getstate__(self):
        return gws.omit(vars(self), 'rel')

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

    def props(self, user):
        return gws.merge(super().props(user), relatedModelUids=[to.model.uid for to in self.rel.to])

    ##

    def find_relatable_features(self, search, mc):
        features = []
        for to in self.rel.to:
            features.extend(to.model.find_features(search, mc))
        return features

    def new_related_feature(self, from_feature, related_model, mc):
        for to in self.rel.to:
            if related_model == to.model:
                record = gws.FeatureRecord(attributes={})
                return to.model.new_feature_from_record(record, mc)

    ##

    def to_props(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        mc2 = mu.secondary_context(mc)

        if self.attributeType == gws.AttributeType.feature:
            for f in features:
                f2 = f.get(self.name)
                if f2:
                    ps = f2.model.features_to_props([f2], mc2)
                    f.props.attributes[self.name] = ps[0]

        if self.attributeType == gws.AttributeType.featurelist:
            for f in features:
                f.props.attributes[self.name] = []
                for f2 in f.get(self.name, []):
                    ps = f2.model.features_to_props([f2], mc2)
                    f.props.attributes[self.name].append(ps[0])

    def from_props(self, features, mc):
        if mc.relDepth >= mc.maxDepth:
            return

        mc2 = mu.secondary_context(mc)
        uid_to_model = {to.model.uid: to.model for to in self.rel.to}
        as_list = self.attributeType == gws.AttributeType.featurelist

        for f in features:
            if as_list:
                f.set(self.name, [])

            val = f.props.attributes.get(self.name)
            if not val:
                continue
            if not isinstance(val, list):
                val = [val]

            for p in val:
                if isinstance(p, dict):
                    p = gws.FeatureProps(**p)
                model = uid_to_model.get(p.modelUid)
                if model:
                    related = gws.first(model.features_from_props([p], mc2))
                    if as_list:
                        f.get(self.name).append(related)
                    else:
                        f.set(self.name, related)

    ##

    def uid_and_key_for_uids(
            self,
            model: gws.IDatabaseModel,
            key_column: sa.Column,
            uids: t.Iterable[gws.ModelKey],
            mc: gws.ModelContext
    ) -> set[tuple[gws.ModelKey, gws.ModelKey]]:

        vs = set(v for v in uids if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(model.uid_column().in_(vs))
        return set((uid, key) for uid, key in self.model.execute(sql, mc))

    def uid_and_key_for_keys(
            self,
            model: gws.IDatabaseModel,
            key_column: sa.Column,
            keys: t.Iterable[gws.ModelKey],
            mc: gws.ModelContext
    ) -> set[tuple[gws.ModelKey, gws.ModelKey]]:

        vs = set(v for v in keys if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(key_column.in_(vs))
        return set((uid, key) for uid, key in self.model.execute(sql, mc))

    def update_uid_and_key(
            self,
            model: gws.IDatabaseModel,
            key_column: sa.Column,
            uid_and_key: t.Iterable[tuple[gws.ModelKey, gws.ModelKey]],
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
            model: gws.IDatabaseModel,
            key_column: sa.Column,
            uids: t.Iterable[gws.ModelKey],
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

    def feature_map(
            self,
            model: gws.IDatabaseModel,
            search: gws.SearchQuery,
            mc: gws.ModelContext
    ) -> dict[gws.ModelKey, gws.IFeature]:
        return {
            f.uid(): f
            for f in model.find_features(search, mu.secondary_context(mc))
        }

    def get_related(
            self,
            model: gws.IDatabaseModel,
            uids: t.Iterable[gws.ModelKey],
            mc: gws.ModelContext
    ) -> list[gws.IFeature]:
        return model.get_features(uids, mu.secondary_context(mc))

    def col_or_uid(self, model, cfg):
        return model.column(cfg) if cfg else model.uid_column()


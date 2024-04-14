"""Related Multi Feature List field

Represents a 1:M relationship betweens a "parent" and multiple "child" tables ::

    +---------+         +------------+
    | parent  |         | child 1    |
    +---------+         +------------+
    | key     |-------<<| parent_key |
    |         |         +------------+
    |         |
    |         |         +------------+
    |         |         | child 2    |
    |         |         +------------+
    |         |-------<<| parent_key |
    |         |         +------------+
    |         |
    |         |         +------------+
    |         |         | child 3    |
    |         |         +------------+
    |         |-------<<| parent_key |
    +---------+         +------------+



"""

import gws
import gws.base.database.model
import gws.base.model.util as mu
import gws.base.model.related_field as related_field
import gws.lib.sa as sa
import gws.types as t

gws.ext.new.modelField('relatedMultiFeatureList')


class RelatedItem(gws.Data):
    toModel: str
    toColumn: str


class Config(related_field.Config):
    fromColumn: str = ''
    """key column in this table, primary key by default"""
    related: list[RelatedItem]
    """related models and keys"""


class Props(related_field.Props):
    pass


class Object(related_field.Object):
    attributeType = gws.AttributeType.featurelist

    def configure_relationship(self):
        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                key=self.column_or_uid(self.model, self.cfg('fromColumn')),
                uid=self.model.uid_column(),
            ),
            tos=[]
        )

        for c in self.cfg('related'):
            to_mod = self.get_model(c.toModel)
            self.rel.tos.append(related_field.RelRef(
                model=to_mod,
                table=to_mod.table(),
                key=to_mod.column(c.toColumn),
                uid=to_mod.uid_column(),
            ))

    ##

    def before_create_related(self, to_feature, mc):
        for feature in to_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(self.rel.src.model, self.rel.src.key, feature.uid(), mc)
                for to in self.rel.tos:
                    if to_feature.model == to.model:
                        to_feature.record.attributes[to.key.name] = key
                        return

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        for to in self.rel.tos:
            sql = sa.select(
                to.uid,
                self.rel.src.uid,
            ).select_from(
                to.table.join(
                    self.rel.src.table, self.rel.src.key.__eq__(to.key)
                )
            ).where(
                self.rel.src.uid.in_(uid_to_f)
            )

            r_to_uids = {}
            for r, u in self.model.execute(sql, mc):
                r_to_uids.setdefault(str(r), []).append(str(u))

            for to_feature in to.model.get_features(r_to_uids, mu.secondary_context(mc)):
                for uid in r_to_uids.get(to_feature.uid(), []):
                    feature = uid_to_f.get(uid)
                    feature.get(self.name).append(to_feature)

    def after_create(self, feature, mc):
        key = self.key_for_uid(self.model, self.rel.src.key, feature.insertedPrimaryKey, mc)
        self.after_write(feature, key, mc)

    def after_update(self, feature, mc):
        key = self.key_for_uid(self.model, self.rel.src.key, feature.uid(), mc)
        self.after_write(feature, key, mc)

    def after_write(self, feature: gws.Feature, key, mc: gws.ModelContext):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        for to in self.rel.tos:
            if not mc.user.can_edit(to.model):
                continue

            cur_uids = self.to_uids_for_key(to, key, mc)

            new_uids = set(
                to_feature.uid()
                for to_feature in feature.get(self.name, [])
                if to_feature.model == to.model
            )

            ins_uids = new_uids - cur_uids
            if ins_uids:
                sql = sa.update(to.table).values({to.key.name: key}).where(to.uid.in_(ins_uids))
                to.model.execute(sql, mc)

            self.drop_links(to, cur_uids - new_uids, mc)

    def before_delete(self, feature, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        key = self.key_for_uid(self.model, self.rel.src.key, feature.uid(), mc)
        setattr(mc, f'_DELETED_KEY_{self.uid}', key)

    def after_delete(self, features, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        key = getattr(mc, f'_DELETED_KEY_{self.uid}')

        for to in self.rel.tos:
            if not mc.user.can_edit(to.model):
                continue
            cur_uids = self.to_uids_for_key(to, key, mc)
            self.drop_links(to, cur_uids, mc)

    def to_uids_for_key(self, to: related_field.RelRef, key, mc):
        sql = sa.select(to.uid).where(to.key.__eq__(key))
        return set(str(u[0]) for u in to.model.execute(sql, mc))

    def drop_links(self, to: related_field.RelRef, to_uids, mc):
        if not to_uids:
            return
        if self.rel.deleteCascade:
            sql = sa.delete(to.table).where(to.uid.in_(to_uids))
        else:
            sql = sa.update(to.table).values({to.key.name: None}).where(to.uid.in_(to_uids))
        to.model.execute(sql, mc)

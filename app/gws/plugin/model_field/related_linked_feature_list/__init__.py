"""Related Linked Feature List field

Represents an M:N relationship between two models via a link table ("associative entity")::

    +---------+         +---------------+         +---------+
    | table A |         | link table    |         | table B |
    +---------+         +---------------+         +---------+
    | key_a   |-------<<| a           b |>>-------| key_b   |
    +---------+         +---------------+         +---------+

"""

import gws
import gws.base.database.model
import gws.base.model.util as mu
import gws.base.model.field
import gws.base.model.related_field as related_field
import gws.lib.sa as sa
import gws.types as t

gws.ext.new.modelField('relatedLinkedFeatureList')


class Config(related_field.Config):
    fromColumn: str = ''
    """key column in this table, primary key by default"""
    toModel: str
    """related model"""
    toColumn: str = ''
    """key column in the related table, primary key by default"""
    linkTableName: str
    """link table name"""
    linkFromColumn: str
    """link key column for this model"""
    linkToColumn: str
    """link key column for the related model"""


class Props(related_field.Props):
    pass


class Object(related_field.Object):
    attributeType = gws.AttributeType.featurelist

    def configure_relationship(self):
        to_mod = self.get_model(self.cfg('toModel'))
        link_tab = self.model.provider.table(self.cfg('linkTableName'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                key=self.column_or_uid(self.model, self.cfg('fromColumn')),
                uid=self.model.uid_column(),
            ),
            tos=[
                related_field.RelRef(
                    model=to_mod,
                    table=to_mod.table(),
                    key=self.column_or_uid(to_mod, self.cfg('toColumn')),
                    uid=to_mod.uid_column(),
                )
            ],
            link=related_field.Link(
                table=link_tab,
                fromKey=self.model.provider.column(link_tab, self.cfg('linkFromColumn')),
                toKey=self.model.provider.column(link_tab, self.cfg('linkToColumn')),
            )
        )
        self.rel.to = self.rel.tos[0]

    ##

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        sql = sa.select(
            self.rel.to.uid,
            self.rel.src.uid,
        ).select_from(
            self.rel.to.table.join(
                self.rel.link.table, self.rel.link.toKey.__eq__(self.rel.to.key)
            ).join(
                self.rel.src.table, self.rel.link.fromKey.__eq__(self.rel.src.key)
            )
        ).where(
            self.rel.src.uid.in_(uid_to_f)
        )

        r_to_uids = {}
        for r, u in self.model.execute(sql, mc):
            r_to_uids.setdefault(str(r), []).append(str(u))

        for to_feature in self.rel.to.model.get_features(r_to_uids, mu.secondary_context(mc)):
            for uid in r_to_uids.get(to_feature.uid(), []):
                feature = uid_to_f.get(uid)
                feature.get(self.name).append(to_feature)

    def after_create_related(self, to_feature, mc):
        if to_feature.model != self.rel.to.model:
            return

        right_key = self.key_for_uid(self.rel.to.model, self.rel.to.key, to_feature.insertedPrimaryKey, mc)
        new_links = set()

        for feature in to_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(self.rel.src.model, self.rel.src.key, feature.uid(), mc)
                new_links.add((key, right_key))

        self.create_links(new_links, mc)

    def after_create(self, feature, mc):
        key = self.key_for_uid(self.model, self.rel.src.key, feature.insertedPrimaryKey, mc)
        self.after_write(feature, key, mc)

    def after_update(self, feature, mc):
        key = self.key_for_uid(self.model, self.rel.src.key, feature.uid(), mc)
        self.after_write(feature, key, mc)

    def after_write(self, feature, key, mc: gws.ModelContext):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        cur_links = self.get_links([key], mc)
        to_uids = set(
            to_feature.uid()
            for to_feature in feature.get(self.name, [])
        )

        sql = sa.select(self.rel.to.uid, self.rel.to.key).where(self.rel.to.uid.in_(to_uids))
        r_uid_to_key = {str(u): k for u, k in self.rel.to.model.execute(sql, mc)}

        new_links = set()

        for to_feature in feature.get(self.name, []):
            right_key = r_uid_to_key.get(to_feature.uid())
            new_links.add((key, right_key))

        self.create_links(new_links - cur_links, mc)
        self.delete_links(cur_links - new_links, mc)

    def get_links(self, left_keys, mc):
        sel = sa.select(
            self.rel.link.fromKey,
            self.rel.link.toKey,
        ).where(
            self.rel.link.fromKey.in_(left_keys)
        )
        return set((lk, rk) for lk, rk in self.model.execute(sel, mc))

    def create_links(self, links, mc):
        sql = sa.insert(self.rel.link.table)
        values = [
            {
                self.rel.link.fromKey.name: lk,
                self.rel.link.toKey.name: rk
            }
            for lk, rk in links
        ]
        if values:
            self.model.execute(sql, mc, values)

    def delete_links(self, links, mc):
        for lk, rk in links:
            sql = sa.delete(
                self.rel.link.table
            ).where(
                self.rel.link.fromKey.__eq__(lk),
                self.rel.link.toKey.__eq__(rk)
            )
            self.model.execute(sql, mc)

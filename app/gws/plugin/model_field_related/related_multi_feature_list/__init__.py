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
                column=self.col_or_uid(self.model, self.cfg('fromColumn')),
                uid=self.model.uid_column(),
            ),
            to=[]
        )

        for c in self.cfg('related'):
            to_mod = t.cast(gws.IDatabaseModel, self.root.get(c.toModel))
            self.rel.to.append(related_field.RelRef(
                model=to_mod,
                table=to_mod.table(),
                column=to_mod.column(c.toColumn),
                uid=to_mod.uid_column(),
            ))

    ##

    def new_related_feature(self, from_feature, related_model, mc):
        uid_and_key = self.uid_and_key_for_uids(
            self.model,
            self.rel.src.column,
            [from_feature.uid()],
            mc
        )

        key = gws.first(k for _, k in uid_and_key)

        for to in self.rel.to:
            if related_model == to.model:
                record = gws.FeatureRecord(attributes={to.column.name: key})
                return to.model.new_feature_from_record(record, mc)

    def before_select(self, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return
        mc.dbSelect.columns.append(self.rel.src.column)

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        for to in self.rel.to:
            sql = sa.select(
                to.uid,
                self.rel.src.uid,
            ).select_from(
                to.table.join(
                    self.rel.src.table, to.column.__eq__(self.rel.src.column)
                )
            ).where(
                self.rel.src.uid.in_(uid_to_f)
            )

            r_to_u = gws.collect(self.model.execute(sql, mc))

            related = self.get_related(
                to.model,
                r_to_u,
                mc
            )

            for rf in related:
                for uid in r_to_u.get(rf.uid()):
                    uid_to_f.get(uid).get(self.name).append(rf)

    def after_create(self, features, mc):
        self.after_write(features, mc)

    def after_update(self, features, mc):
        self.after_write(features, mc)

    def after_write(self, features, mc: gws.ModelContext):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        uid_to_key = dict(self.uid_and_key_for_uids(
            self.model,
            self.rel.src.column,
            [f.uid() for f in features],
            mc
        ))

        for to in self.rel.to:
            if not mc.user.can_edit(to.model):
                continue

            cur_pairs = self.uid_and_key_for_keys(
                to.model,
                to.column,
                uid_to_key.values(),
                mc
            )

            new_pairs = set(
                (related.uid(), uid_to_key.get(f.uid()))
                for f in features
                for related in f.get(self.name, [])
                if related.model == to.model
            )

            self.update_uid_and_key(
                to.model,
                to.column,
                new_pairs - cur_pairs,
                mc
            )

            self.drop_uid_and_key(
                to.model,
                to.column,
                set(r for r, _ in cur_pairs) - set(r for r, _ in new_pairs),
                False,
                mc
            )

    def before_delete(self, features, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        uid_to_key = dict(self.uid_and_key_for_uids(
            self.model,
            self.rel.src.column,
            [f.uid() for f in features],
            mc
        ))
        setattr(mc, f'_uid_to_key_{self.uid}', uid_to_key)

    def after_delete(self, features, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        uid_to_key = getattr(mc, f'_uid_to_key_{self.uid}')

        for to in self.rel.to:
            if not mc.user.can_edit(to.model):
                continue

            cur_pairs = self.uid_and_key_for_keys(
                to.model,
                to.column,
                uid_to_key.values(),
                mc
            )
            self.drop_uid_and_key(
                to.model,
                to.column,
                set(r for r, _ in cur_pairs),
                False,
                mc
            )

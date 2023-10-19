"""Related Feature field

Represents a child->parent M:1 relationship to another model::

    +-------------+         +--------------+
    | model       |         | toModel      |
    +-------------+         +--------------+
    | fromKey     |-------->| toKey        |
    +-------------+         +--------------+

The value of the field is the parent feature.
"""

import gws
import gws.base.database.model
import gws.base.model.related_field as related_field
import gws.lib.sa as sa
import gws.types as t

gws.ext.new.modelField('relatedFeature')


class Config(related_field.Config):
    fromColumn: str
    """foreign key column in this table"""
    toModel: str
    """related model"""
    toColumn: str = ''
    """key column in the related model, primary key by default"""


class Props(related_field.Props):
    pass


class Object(related_field.Object):
    attributeType = gws.AttributeType.feature

    def configure_relationship(self):
        to_mod = t.cast(gws.IDatabaseModel, self.root.get(self.cfg('toModel')))

        self.rel = related_field.Relationship(

            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                column=self.model.column(self.cfg('fromColumn')),
                uid=self.model.uid_column(),
            ),

            to=[
                related_field.RelRef(
                    model=to_mod,
                    table=to_mod.table(),
                    column=self.col_or_uid(to_mod, self.cfg('toColumn')),
                    uid=to_mod.uid_column(),
                )
            ],
        )

    ##

    def do_init_with_record(self, feature, mc):
        self.load_related_from_record([feature], mc)

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return
        self.load_related_from_record(features, mc)

    def before_create(self, features, mc):
        self.before_write(features, mc)

    def before_update(self, features, mc):
        self.before_write(features, mc)

    def before_write(self, features: list[gws.IFeature], mc: gws.ModelContext):
        if not mc.user.can_write(self):
            return

        for f in features:
            f.record.attributes[self.rel.src.column.name] = None

        related_uid_to_f = [
            (f.get(self.name).uid(), f)
            for f in features
            if f.get(self.name)
        ]
        related_uid_and_key = self.uid_and_key_for_uids(
            self.rel.to[0].model,
            self.rel.to[0].column,
            [r for r, _ in related_uid_to_f],
            mc
        )
        for rel_uid, f in related_uid_to_f:
            keys = [k for r, k in related_uid_and_key if r == rel_uid]
            if keys:
                f.record.attributes[self.rel.src.column.name] = keys[0]

    def load_related_from_record(self, features, mc):
        uid_to_f = {f.uid(): f for f in features}

        sql = sa.select(
            self.rel.to[0].uid,
            self.rel.src.uid,
        ).select_from(
            self.rel.to[0].table.join(
                self.rel.src.table, self.rel.to[0].column.__eq__(self.rel.src.column)
            )
        ).where(
            self.rel.src.uid.in_(uid_to_f)
        )

        r_to_u = gws.collect(self.model.execute(sql, mc))

        related = self.get_related(
            self.rel.to[0].model,
            r_to_u,
            mc
        )

        for rf in related:
            for uid in r_to_u.get(rf.uid()):
                uid_to_f.get(uid).set(self.name, rf)

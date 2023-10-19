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
        to_mod = t.cast(gws.IDatabaseModel, self.root.get(self.cfg('toModel')))

        lt = self.model.provider.table(self.cfg('linkTableName'))
        fc = self.model.provider.column(lt, self.cfg('linkFromColumn'))
        tc = self.model.provider.column(lt, self.cfg('linkToColumn'))

        self.rel = related_field.Relationship(

            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                column=self.col_or_uid(self.model, self.cfg('fromColumn')),
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

            link=related_field.Link(
                table=lt,
                fromColumn=fc,
                toColumn=tc
            )
        )

    ##

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        sql = sa.select(
            self.rel.to[0].uid,
            self.rel.src.uid,
        ).select_from(
            self.rel.to[0].table.join(
                self.rel.link.table, self.rel.link.toColumn.__eq__(self.rel.to[0].column)
            ).join(
                self.rel.src.table, self.rel.link.fromColumn.__eq__(self.rel.src.column)
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

        cur_links = self.get_links(uid_to_key.values(), mc)

        related_uids = set(
            related.uid()
            for f in features
            for related in f.get(self.name, [])
        )

        related_uid_to_key = dict(self.uid_and_key_for_uids(
            self.rel.to[0].model,
            self.rel.to[0].column,
            related_uids,
            mc
        ))

        new_links = set()

        for f in features:
            key = uid_to_key.get(f.uid())
            for related in f.get(self.name, []):
                rel_key = related_uid_to_key.get(related.uid())
                new_links.add((key, rel_key))

        self.create_links(new_links - cur_links, mc)
        self.delete_links(cur_links - new_links, mc)

    def get_links(self, left_keys, mc):
        sel = sa.select(
            self.rel.link.fromColumn,
            self.rel.link.toColumn,
        ).where(
            self.rel.link.fromColumn.in_(left_keys)
        )
        return set((lk, rk) for lk, rk in self.model.execute(sel, mc))

    def create_links(self, links, mc):
        sql = sa.insert(self.rel.link.table)
        values = [
            {
                self.rel.link.fromColumn.name: lk,
                self.rel.link.toColumn.name: rk
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
                self.rel.link.fromColumn.__eq__(lk),
                self.rel.link.toColumn.__eq__(rk)
            )
            self.model.execute(sql, mc)

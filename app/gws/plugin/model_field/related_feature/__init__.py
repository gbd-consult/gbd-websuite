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
import gws.base.model
import gws.base.model.related_field as related_field
import gws.lib.sa as sa

gws.ext.new.modelField('relatedFeature')


class Config(related_field.Config):
    """Configuration for related feature field."""

    fromColumn: str
    """Foreign key column in this table."""
    toModel: str
    """Related model."""
    toColumn: str = ''
    """Key column in the related model, primary key by default."""


class Props(related_field.Props):
    pass


class Object(related_field.Object):
    attributeType = gws.AttributeType.feature

    def configure_relationship(self):
        to_mod = self.get_model(self.cfg('toModel'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                key=self.model.column(self.cfg('fromColumn')),
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
        )
        self.rel.to = self.rel.tos[0]

    ##

    def do_init(self, feature, mc):
        key = feature.record.attributes.get(self.rel.src.key.name)
        if key:
            to_uids = self.uids_for_key(self.rel.to, key, mc)
            to_features = self.rel.to.model.get_features(
                to_uids,
                gws.base.model.secondary_context(mc),
            )
            if to_features:
                feature.attributes[self.name] = to_features[0]

    def after_create_related(self, to_feature, mc):
        if to_feature.model != self.rel.to.model:
            return

        for feature in to_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(
                    self.rel.to.model,
                    self.rel.to.key,
                    to_feature.insertedPrimaryKey,
                    mc,
                )
                if key:
                    self.update_key_for_uids(
                        self.model,
                        self.rel.src.key,
                        [feature.uid()],
                        key,
                        mc,
                    )

    def uids_for_key(self, rel: related_field.RelRef, key, mc):
        sql = sa.select(rel.uid).where(rel.key.__eq__(key))
        with rel.model.db.connect() as conn:
            return set(str(u) for u in conn.execute(sql))

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        uid_to_f = {f.uid(): f for f in features}

        sql = (
            sa.select(
                self.rel.to.uid,
                self.rel.src.uid,
            )
            .select_from(
                self.rel.to.table.join(
                    self.rel.src.table,
                    self.rel.src.key.__eq__(self.rel.to.key),
                ),
            )
            .where(self.rel.src.uid.in_(uid_to_f))
        )

        r_to_uids = {}
        with self.model.db.connect() as conn:
            for r, u in conn.execute(sql):
                r_to_uids.setdefault(str(r), []).append(str(u))

        for to_feature in self.rel.to.model.get_features(
            r_to_uids,
            gws.base.model.secondary_context(mc),
        ):
            for uid in r_to_uids.get(to_feature.uid(), []):
                feature = uid_to_f.get(uid)
                feature.set(self.name, to_feature)

    def before_create(self, feature, mc):
        self.before_write(feature, mc)

    def before_update(self, feature, mc):
        self.before_write(feature, mc)

    def before_write(self, feature: gws.Feature, mc: gws.ModelContext):
        if not mc.user.can_write(self):
            return

        if feature.has(self.name):
            key = None
            to_feature = feature.get(self.name)
            if to_feature:
                key = self.key_for_uid(
                    self.rel.to.model,
                    self.rel.to.key,
                    to_feature.uid(),
                    mc,
                )
            feature.record.attributes[self.rel.src.key.name] = key

"""Related feature field.

Represents a child->parent M:1 relationship to another model::

    +-------------+         +--------------+
    | model       |         | toModel      |
    +-------------+         +--------------+
    | fromColumn  |-------->| toColumn     |
    +-------------+         +--------------+

The value of the field is the parent feature. ``fromColumn`` is the foreign
key column in this model's table, ``toColumn`` the key column in the related
model, by default its primary key. When a feature is written, the key of the
selected parent feature is stored in ``fromColumn``. When a parent feature is
created together with a child feature, the child's ``fromColumn`` is set to
the new parent's key. Without a configured widget, the field uses a
``featureSelect`` widget.

Example::

    fields+ {
        name "category"
        type "relatedFeature"
        fromColumn "category_id"
        toModel "model_category"
        toColumn "id"
        widget.type "featureSelect"
    }
"""

import gws
import gws.base.model
import gws.base.model.related_field as related_field
import gws.lib.sa as sa


@gws.ext.config.modelField('relatedFeature')
class Config(related_field.Config):
    """Field referring to a feature in another model."""

    fromColumn: str
    """Foreign key column in this table."""
    toModel: str
    """UID of the related model."""
    toColumn: str = ''
    """Key column in the related model, primary key by default."""


@gws.ext.props.modelField('relatedFeature')
class Props(related_field.Props):
    pass


@gws.ext.object.modelField('relatedFeature')
class Object(related_field.Object):
    """Related feature field object."""

    attributeType = gws.AttributeType.feature

    def configure_relationship(self):
        dst_mod = self.get_model(self.cfg('toModel'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                keyName=self.model.column(self.cfg('fromColumn')).name,
            ),
            dstList=[
                related_field.RelRef(
                    model=dst_mod,
                    keyName=dst_mod.column(self.cfg('toColumn') or dst_mod.uidName).name,
                )
            ],
        )
        self.rel.dst = self.rel.dstList[0]

    ##

    def do_init(self, feature, mc):
        key = feature.record.attributes.get(self.rel.src.keyName)
        if key:
            dst_uids = self.uids_for_key(self.rel.dst, key, mc)
            dst_features = self.rel.dst.model.get_features(
                dst_uids,
                gws.base.model.secondary_context(mc),
            )
            if dst_features:
                feature.attributes[self.name] = dst_features[0]

    def after_create_related(self, dst_feature, mc):
        if dst_feature.model != self.rel.dst.model:
            return

        for feature in dst_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(
                    self.rel.dst.model,
                    self.key_column(self.rel.dst),
                    dst_feature.insertedPrimaryKey,
                    mc,
                )
                if key:
                    sql = (
                        sa.update(self.model.table())
                        .values({self.rel.src.keyName: key})
                        .where(self.model.uid_equals(feature.uid()))
                    )
                    with self.model.db.begin() as conn:
                        conn.execute(sql)

    def uids_for_key(self, rel: related_field.RelRef, key, mc):
        """Find the uids of the features whose key column has the given value.

        Args:
            rel: The relationship side to query.
            key: The key value.
            mc: The model context.

        Returns:
            A set of feature uids as strings.
        """
        sql = sa.select(rel.model.uid_column()).where(self.key_column(rel) == key)
        with rel.model.db.begin() as conn:
            return set(str(u) for u in conn.execute(sql))

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        uid_to_f = {f.uid(): f for f in features}

        sql = (
            sa.select(
                self.rel.dst.model.uid_column(),
                self.rel.src.model.uid_column(),
            )
            .select_from(
                self.rel.dst.model.table().join(
                    self.rel.src.model.table(),
                    self.key_column(self.rel.src) == self.key_column(self.rel.dst),
                ),
            )
            .where(self.rel.src.model.uid_equals(uid_to_f))
        )

        r_to_uids = {}
        with self.model.db.begin() as conn:
            for r, u in conn.execute(sql):
                r_to_uids.setdefault(str(r), []).append(str(u))

        for dst_feature in self.rel.dst.model.get_features(
            r_to_uids,
            gws.base.model.secondary_context(mc),
        ):
            for uid in r_to_uids.get(dst_feature.uid(), []):
                feature = uid_to_f.get(uid)
                feature.set(self.name, dst_feature)

    def before_create(self, feature, mc):
        self.before_write(feature, mc)

    def before_update(self, feature, mc):
        self.before_write(feature, mc)

    def before_write(self, feature: gws.Feature, mc: gws.ModelContext):
        """Write the key of the related feature to the foreign key column of the record.

        Does nothing if the user may not write the field or the feature has no value
        for it. An empty value clears the foreign key.

        Args:
            feature: The feature being created or updated.
            mc: The model context.
        """
        if not mc.user.can_write(self):
            return

        if feature.has(self.name):
            key = None
            dst_feature = feature.get(self.name)
            if dst_feature:
                key = self.key_for_uid(
                    self.rel.dst.model,
                    self.key_column(self.rel.dst),
                    dst_feature.uid(),
                    mc,
                )
            feature.record.attributes[self.rel.src.keyName] = key

"""Related multi feature list field.

Represents a 1:M relationship between a "parent" and multiple "child" tables::

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

The value of the field is the list of child features from all child models.
``fromColumn`` is the key column in this model's table, by default its primary
key; each entry in ``related`` names a child model and its foreign key column.

When a feature is written, child features in the list get their foreign key
set to the parent key. Child features no longer in the list are unlinked by
clearing their foreign key. When a parent feature is deleted, its children are
unlinked in the same way. Child models the user may not edit are skipped.
When a child feature is created together with a parent feature, its foreign
key is set to the parent key. Without a configured widget, the field uses a
``featureList`` widget.

Example::

    fields+ {
        name "documents"
        type "relatedMultiFeatureList"
        related [
            { toModel "model_photo" toColumn "parent_id" }
            { toModel "model_report" toColumn "parent_id" }
        ]
    }
"""

import gws
import gws.base.model
import gws.base.model.related_field as related_field
import gws.lib.sa as sa


class RelatedItem(gws.Data):
    """Related model and its key column."""

    toModel: str
    """UID of the related model."""
    toColumn: str
    """Foreign key column in the related model."""


@gws.ext.config.modelField('relatedMultiFeatureList')
class Config(related_field.Config):
    """Field listing related features from several models."""

    fromColumn: str = ''
    """Key column in this table, primary key by default."""
    related: list[RelatedItem]
    """Related models and keys."""


@gws.ext.props.modelField('relatedMultiFeatureList')
class Props(related_field.Props):
    pass


@gws.ext.object.modelField('relatedMultiFeatureList')
class Object(related_field.Object):
    """Related multi feature list field object."""

    attributeType = gws.AttributeType.featurelist

    def configure_relationship(self):
        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                keyName=self.model.column(self.cfg('fromColumn') or self.model.uidName).name,
            ),
            dstList=[],
        )

        for c in self.cfg('related'):
            dst_mod = self.get_model(c.toModel)
            self.rel.dstList.append(
                related_field.RelRef(
                    model=dst_mod,
                    keyName=dst_mod.column(c.toColumn).name,
                )
            )

    ##

    def before_create_related(self, dst_feature, mc):
        for feature in dst_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(
                    self.rel.src.model,
                    self.key_column(self.rel.src),
                    feature.uid(),
                    mc,
                )
                for dst in self.rel.dstList:
                    if dst_feature.model == dst.model:
                        dst_feature.record.attributes[dst.keyName] = key
                        return

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        for dst in self.rel.dstList:
            sql = (
                sa.select(
                    dst.model.uid_column(),
                    self.rel.src.model.uid_column(),
                )
                .select_from(
                    dst.model.table().join(
                        self.rel.src.model.table(),
                        self.key_column(self.rel.src) == self.key_column(dst),
                    ),
                )
                .where(
                    self.rel.src.model.uid_equals(uid_to_f),
                )
            )

            r_to_uids = {}
            with self.model.db.begin() as conn:
                for r, u in conn.execute(sql):
                    r_to_uids.setdefault(str(r), []).append(str(u))

            for dst_feature in dst.model.get_features(
                r_to_uids,
                gws.base.model.secondary_context(mc),
            ):
                for uid in r_to_uids.get(dst_feature.uid(), []):
                    feature = uid_to_f.get(uid)
                    feature.get(self.name).append(dst_feature)

    def after_create(self, feature, mc):
        key = self.key_for_uid(
            self.model,
            self.key_column(self.rel.src),
            feature.insertedPrimaryKey,
            mc,
        )
        self.after_write(feature, key, mc)

    def after_update(self, feature, mc):
        key = self.key_for_uid(
            self.model,
            self.key_column(self.rel.src),
            feature.uid(),
            mc,
        )
        self.after_write(feature, key, mc)

    def after_write(self, feature: gws.Feature, key, mc: gws.ModelContext):
        """Link the child features in the field value to a written feature and unlink the others.

        Does nothing if the user may not write the field or the maximum relation
        depth is reached. Child models the user may not edit are skipped.

        Args:
            feature: The created or updated feature.
            key: Value of the key column of the feature.
            mc: The model context.
        """
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        for dst in self.rel.dstList:
            if not mc.user.can_edit(dst.model):
                continue

            cur_uids = self.dst_uids_for_key(dst, key, mc)

            # fmt: off
            new_uids = set(
                dst_feature.uid() 
                for dst_feature in feature.get(self.name, []) 
                if dst_feature.model == dst.model
            )
            # fmt: on

            ins_uids = new_uids - cur_uids
            if ins_uids:
                sql = (
                    sa.update(dst.model.table())
                    .values(
                        {dst.keyName: key},
                    )
                    .where(
                        dst.model.uid_equals(ins_uids),
                    )
                )
                with dst.model.db.begin() as conn:
                    conn.execute(sql)

            self.drop_links(dst, cur_uids - new_uids, mc)

    def before_delete(self, feature, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        key = self.key_for_uid(
            self.model,
            self.key_column(self.rel.src),
            feature.uid(),
            mc,
        )
        setattr(mc, f'_DELETED_KEY_{self.uid}', key)

    def after_delete(self, features, mc):
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        key = getattr(mc, f'_DELETED_KEY_{self.uid}')

        for dst in self.rel.dstList:
            if not mc.user.can_edit(dst.model):
                continue
            cur_uids = self.dst_uids_for_key(dst, key, mc)
            self.drop_links(dst, cur_uids, mc)

    def dst_uids_for_key(self, dst: related_field.RelRef, key, mc):
        """Find the uids of the child features that refer to a parent key.

        Args:
            dst: The child side of the relationship.
            key: The parent key value.
            mc: The model context.

        Returns:
            A set of child feature uids as strings.
        """
        sql = sa.select(dst.model.uid_column()).where(self.key_column(dst) == key)
        with dst.model.db.begin() as conn:
            return set(str(u[0]) for u in conn.execute(sql))

    def drop_links(self, dst: related_field.RelRef, dst_uids, mc):
        """Unlink child features from their parent.

        Clears the foreign key of the child features, or deletes them if the
        relationship has ``deleteCascade`` set.

        Args:
            dst: The child side of the relationship.
            dst_uids: Uids of the child features.
            mc: The model context.
        """
        if not dst_uids:
            return
        if self.rel.deleteCascade:
            sql = sa.delete(dst.model.table()).where(dst.model.uid_equals(dst_uids))
        else:
            sql = (
                sa.update(dst.model.table())
                .values(
                    {dst.keyName: None},
                )
                .where(
                    dst.model.uid_equals(dst_uids),
                )
            )
        with dst.model.db.begin() as conn:
            conn.execute(sql)

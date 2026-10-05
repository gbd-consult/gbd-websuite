"""Related linked feature list field.

Represents an M:N relationship between two models via a link table ("associative entity")::

    +------------+         +-------------------------------+         +-----------+
    | this model |         | linkTableName                 |         | toModel   |
    +------------+         +-------------------------------+         +-----------+
    | fromColumn |-------<<| linkFromColumn   linkToColumn |>>-------| toColumn  |
    +------------+         +-------------------------------+         +-----------+

The value of the field is the list of linked features of the related model.
``fromColumn`` and ``toColumn`` default to the primary keys. The link table
must be in the same database as this model.

When a feature is written, the rows of the link table are synchronized with
the field value: missing links are inserted, links to features no longer in
the list are deleted. The linked features themselves are not changed. When a
related feature is created together with features of this model, links to
them are inserted. Without a configured widget, the field uses a
``featureList`` widget.

Example::

    fields+ {
        name "tags"
        type "relatedLinkedFeatureList"
        toModel "model_tag"
        linkTableName "edit.poi_tag"
        linkFromColumn "poi_id"
        linkToColumn "tag_id"
    }
"""

import gws
import gws.base.database.model
import gws.base.model
import gws.base.model.related_field as related_field
import gws.lib.sa as sa


@gws.ext.config.modelField('relatedLinkedFeatureList')
class Config(related_field.Config):
    """Field listing features of another model, related via a link table."""

    fromColumn: str = ''
    """Key column in this table, primary key by default."""
    toModel: str
    """UID of the related model."""
    toColumn: str = ''
    """Key column in the related table, primary key by default."""
    linkTableName: str
    """Link table name."""
    linkFromColumn: str
    """Column in the link table that refers to this model."""
    linkToColumn: str
    """Column in the link table that refers to the related model."""


@gws.ext.props.modelField('relatedLinkedFeatureList')
class Props(related_field.Props):
    pass


@gws.ext.object.modelField('relatedLinkedFeatureList')
class Object(related_field.Object):
    """Related linked feature list field object."""

    attributeType = gws.AttributeType.featurelist

    def configure_relationship(self):
        dst_mod = self.get_model(self.cfg('toModel'))
        link_tab = self.model.db.table(self.cfg('linkTableName'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                keyName=self.model.column(self.cfg('fromColumn') or self.model.uidName).name,
            ),
            dstList=[
                related_field.RelRef(
                    model=dst_mod,
                    keyName=dst_mod.column(self.cfg('toColumn') or dst_mod.uidName).name,
                )
            ],
            link=related_field.Link(
                tableName=self.cfg('linkTableName'),
                srcKeyName=self.model.db.column(link_tab, self.cfg('linkFromColumn')).name,
                dstKeyName=self.model.db.column(link_tab, self.cfg('linkToColumn')).name,
            ),
        )
        self.rel.dst = self.rel.dstList[0]

    ##

    def after_select(self, features, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        for f in features:
            f.set(self.name, [])

        uid_to_f = {f.uid(): f for f in features}

        sql = (
            sa.select(
                self.rel.dst.model.uid_column(),
                self.rel.src.model.uid_column(),
            )
            .select_from(
                self.rel.dst.model.table().join(
                    self.link_table(),
                    self.link_column(self.rel.link.dstKeyName) == self.key_column(self.rel.dst),
                ).join(
                    self.rel.src.model.table(),
                    self.link_column(self.rel.link.srcKeyName) == self.key_column(self.rel.src),
                )
            )
            .where(
                self.rel.src.model.uid_equals(uid_to_f),
            )
        )

        r_to_uids = {}
        with self.model.db.connect() as conn:
            for r, u in conn.execute(sql):
                r_to_uids.setdefault(str(r), []).append(str(u))

        for dst_feature in self.rel.dst.model.get_features(
            r_to_uids,
            gws.base.model.secondary_context(mc),
        ):
            for uid in r_to_uids.get(dst_feature.uid(), []):
                feature = uid_to_f.get(uid)
                feature.get(self.name).append(dst_feature)

    def after_create_related(self, dst_feature, mc):
        if dst_feature.model != self.rel.dst.model:
            return

        right_key = self.key_for_uid(
            self.rel.dst.model,
            self.key_column(self.rel.dst),
            dst_feature.insertedPrimaryKey,
            mc,
        )
        new_links = set()

        for feature in dst_feature.createWithFeatures:
            if feature.model == self.model:
                key = self.key_for_uid(
                    self.rel.src.model,
                    self.key_column(self.rel.src),
                    feature.uid(),
                    mc,
                )
                new_links.add((key, right_key))

        self.create_links(new_links, mc)

    def after_create(self, feature, mc):
        key = self.key_for_uid(
            self.model,
            self.key_column(self.rel.src),
            feature.insertedPrimaryKey,
            mc,
        )
        self.after_write(feature, key, mc)

    def after_update(self, feature, mc):
        key = self.key_for_uid(self.model, self.key_column(self.rel.src), feature.uid(), mc)
        self.after_write(feature, key, mc)

    def after_write(self, feature, key, mc: gws.ModelContext):
        """Synchronize the link table with the field value of a written feature.

        Does nothing if the user may not write the field or the maximum relation
        depth is reached.

        Args:
            feature: The created or updated feature.
            key: Value of the key column of the feature.
            mc: The model context.
        """
        if not mc.user.can_write(self) or mc.relDepth >= mc.maxDepth:
            return

        cur_links = self.get_links([key], mc)
        dst_uids = set(dst_feature.uid() for dst_feature in feature.get(self.name, []))

        sql = sa.select(
            self.rel.dst.model.uid_column(),
            self.key_column(self.rel.dst),
        ).where(
            self.rel.dst.model.uid_equals(dst_uids),
        )
        with self.rel.dst.model.db.connect() as conn:
            r_uid_to_key = {str(u): k for u, k in conn.execute(sql)}

        new_links = set()

        for dst_feature in feature.get(self.name, []):
            right_key = r_uid_to_key.get(dst_feature.uid())
            new_links.add((key, right_key))

        self.create_links(new_links - cur_links, mc)
        self.delete_links(cur_links - new_links, mc)

    def get_links(self, left_keys, mc):
        """Read the links of the given keys from the link table.

        Args:
            left_keys: Key values of features of this model.
            mc: The model context.

        Returns:
            A set of ``(src_key, dst_key)`` tuples.
        """
        sql = sa.select(
            self.link_column(self.rel.link.srcKeyName),
            self.link_column(self.rel.link.dstKeyName),
        ).where(
            self.link_column(self.rel.link.srcKeyName).in_(left_keys),
        )
        with self.model.db.connect() as conn:
            return set((lk, rk) for lk, rk in conn.execute(sql))

    def create_links(self, links, mc):
        """Insert links into the link table.

        Args:
            links: ``(src_key, dst_key)`` tuples.
            mc: The model context.
        """
        sql = sa.insert(self.link_table())
        # fmt: off
        values = [
            {self.rel.link.srcKeyName: lk, self.rel.link.dstKeyName: rk} 
            for lk, rk in links
        ]
        # fmt: on
        if values:
            with self.model.db.connect() as conn:
                conn.execute(sql, values)

    def delete_links(self, links, mc):
        """Delete links from the link table.

        Args:
            links: ``(src_key, dst_key)`` tuples.
            mc: The model context.
        """
        with self.model.db.connect() as conn:
            for lk, rk in links:
                sql = sa.delete(
                    self.link_table(),
                ).where(
                    self.link_column(self.rel.link.srcKeyName) == lk,
                    self.link_column(self.rel.link.dstKeyName) == rk,
                )
                conn.execute(sql)

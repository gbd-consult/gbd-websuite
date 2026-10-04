"""Base class for related fields."""

from typing import Optional, Iterable, Any, cast

import gws
import gws.lib.sa as sa

from . import field


class Config(field.Config):
    pass


class Props(field.Props):
    pass


class Link(gws.Data):
    """Link table of a many-to-many relationship."""

    table: sa.Table
    """The link table."""
    fromKey: sa.Column
    """Column in the link table that refers to the source model."""
    toKey: sa.Column
    """Column in the link table that refers to the related model."""


class RelRef(gws.Data):
    """One side of a relationship."""

    model: gws.DatabaseModel
    """The model."""
    table: sa.Table
    """The table of the model."""
    key: sa.Column
    """The column that takes part in the relationship."""
    uid: sa.Column
    """The uid (primary key) column of the model."""


class Relationship(gws.Data):
    """Relationship between the model of a field and related models."""

    src: RelRef
    """The side of the field's own model."""
    to: RelRef
    """The related side, for relationships with a single related model."""
    tos: list[RelRef]
    """The related sides."""
    link: Link
    """The link table, for many-to-many relationships."""
    deleteCascade: bool = False
    """When unlinking related features, delete them instead of clearing their key."""


class Object(field.Object):
    """Base related field.

    Links features of the field's model to features of other database models.
    Provides the relationship description, conversion of related features to and from
    props, and database helpers to read and update the keys of related features.
    Subclasses describe the link in `configure_relationship`.
    """

    model: gws.DatabaseModel
    """The model of this field."""
    rel: Relationship
    """The relationship of this field."""

    def __getstate__(self):
        """Return the object state without the relationship."""
        return gws.u.omit(vars(self), 'rel')

    def post_configure(self):
        self.configure_relationship()

    def activate(self):
        self.configure_relationship()

    def configure_relationship(self):
        """Set up the relationship ``rel``.

        Called after configuration and again on activation, because the relationship
        holds SQLAlchemy objects and is not pickled. The base implementation does nothing.
        """
        pass

    def configure_widget(self):
        if not super().configure_widget():
            if self.attributeType == gws.AttributeType.feature:
                self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='featureSelect')
                return True
            if self.attributeType == gws.AttributeType.featurelist:
                self.widget = self.root.create_shared(gws.ext.object.modelWidget, type='featureList')
                return True

    def get_model(self, uid: str) -> gws.DatabaseModel:
        """Return a model by uid.

        Args:
            uid: The model uid.

        Returns:
            The model.

        Raises:
            gws.ConfigurationError: If the model is not found.
        """
        mod = self.root.get(uid)
        if not mod:
            raise gws.ConfigurationError(f'model {uid!r} not found')
        return cast(gws.DatabaseModel, mod)

    def find_relatable_features(self, search, mc):
        return [
            f
            for to in self.rel.tos
            for f in to.model.find_features(search, mc)
        ]

    def related_field(self, to: RelRef) -> Optional[gws.ModelField]:
        """Find the field of a related model that describes the reverse relationship.

        Args:
            to: The related side of the relationship.

        Returns:
            The field of ``to.model`` whose relationship starts at ``to.model`` and ``to.key``,
            or None if there is no such field.
        """
        for fld in to.model.fields:
            rel2 = cast(Relationship, getattr(fld, 'rel', None))
            if not rel2:
                continue
            if rel2.src.model == to.model and rel2.src.key == to.key:
                return fld

    def do_init_related(self, to_feature, mc):
        our_features = [f for f in to_feature.createWithFeatures if f.model == self.model]
        if not our_features:
            return

        for to in self.rel.tos:
            if to.model == to_feature.model:
                fld = self.related_field(to)
                if fld:
                    if fld.attributeType == gws.AttributeType.feature:
                        to_feature.attributes[fld.name] = our_features[0]
                    if fld.attributeType == gws.AttributeType.featurelist:
                        to_feature.attributes.setdefault(fld.name, []).extend(our_features)

    def related_models(self):
        return [to.model for to in self.rel.tos]

    ##

    def to_props(self, feature, mc):
        if not mc.user.can_read(self) or mc.relDepth >= mc.maxDepth:
            return

        value = feature.get(self.name)
        if not value:
            return
        if not isinstance(value, list):
            value = [value]

        mc2 = gws.base.model.secondary_context(mc)
        res = []

        for v in value:
            related = cast(gws.Feature, v)
            if related:
                p = related.model.feature_to_props(related, mc2)
                if p:
                    res.append(p)

        if self.attributeType == gws.AttributeType.featurelist:
            feature.props.attributes[self.name] = res
        elif res:
            feature.props.attributes[self.name] = res[0]

    def from_props(self, feature, mc):
        if mc.relDepth >= mc.maxDepth:
            return

        value = feature.props.attributes.get(self.name)
        if not value:
            return
        if not isinstance(value, list):
            value = [value]

        mc2 = gws.base.model.secondary_context(mc)
        res = []
        to_model_map: dict[str, gws.Model] = {to.model.uid: to.model for to in self.rel.tos}

        for v in value:
            rel_props = cast(gws.FeatureProps, gws.u.to_data_object(v))
            if rel_props:
                to_model = to_model_map.get(rel_props.modelUid)
                if to_model:
                    related = to_model.feature_from_props(rel_props, mc2)
                    if related:
                        res.append(related)

        if self.attributeType == gws.AttributeType.featurelist:
            feature.set(self.name, res)
        elif res:
            feature.set(self.name, res[0])

    ##

    def key_for_uid(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uid: gws.FeatureUid,
            mc: gws.ModelContext
    ):
        """Return the key value of the feature with the given uid.

        Args:
            model: The model to query.
            key_column: The key column.
            uid: The feature uid.
            mc: The model context.

        Returns:
            The key value, or None if the feature is not found.
        """
        sql = sa.select(key_column).where(model.uid_column().__eq__(uid))
        with model.db.connect() as conn:
            rs = list(conn.execute(sql))
        return rs[0][0] if rs else None

    def update_key_for_uids(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: list[gws.FeatureUid],
            key: Any,
            mc: gws.ModelContext
    ):
        """Set the key column to a value for the features with the given uids.

        Args:
            model: The model to update.
            key_column: The key column.
            uids: The feature uids.
            key: The new key value.
            mc: The model context.
        """

        sql = sa.update(
            model.table()
        ).values({
            key_column: key
        }).where(
            model.uid_column().in_(uids)
        )
        with model.db.connect() as conn:
            conn.execute(sql)

    def uids_to_keys(
            self,
            mc: gws.ModelContext,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: Optional[Iterable[gws.FeatureUid]] = None,
            keys: Optional[Iterable[Any]] = None,
    ):
        """Map feature uids to key values.

        Selects the features either by ``uids`` or, if no uids are given, by ``keys``.
        None values are ignored.

        Args:
            mc: The model context.
            model: The model to query.
            key_column: The key column.
            uids: Feature uids to look up.
            keys: Key values to look up.

        Returns:
            A dict mapping uids (as strings) to key values.
        """

        if uids:
            uids = set(v for v in uids if v is not None)
            sql = sa.select(model.uid_column(), key_column).where(model.uid_column().in_(uids))
        else:
            keys = set(v for v in keys if v is not None)
            sql = sa.select(model.uid_column(), key_column).where(key_column.in_(keys))

        with model.db.connect() as conn:
            return {str(uid): key for uid, key in conn.execute(sql)}

    def uid_and_key_for_uids(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> set[tuple[gws.FeatureUid, gws.FeatureUid]]:
        """Return uid and key pairs for the features with the given uids.

        Args:
            model: The model to query.
            key_column: The key column.
            uids: The feature uids. None values are ignored.
            mc: The model context.

        Returns:
            A set of ``(uid, key)`` tuples.
        """

        vs = set(v for v in uids if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(model.uid_column().in_(vs))
        with model.db.connect() as conn:
            return set((uid, key) for uid, key in conn.execute(sql))

    def uid_and_key_for_keys(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            keys: Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> set[tuple[gws.FeatureUid, gws.FeatureUid]]:
        """Return uid and key pairs for the features with the given key values.

        Args:
            model: The model to query.
            key_column: The key column.
            keys: The key values. None values are ignored.
            mc: The model context.

        Returns:
            A set of ``(uid, key)`` tuples.
        """

        vs = set(v for v in keys if v is not None)
        sql = sa.select(model.uid_column(), key_column).where(key_column.in_(vs))
        with model.db.connect() as conn:
            return set((uid, key) for uid, key in conn.execute(sql))

    def update_uid_and_key(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uid_and_key: Iterable[tuple[gws.FeatureUid, gws.FeatureUid]],
            mc: gws.ModelContext
    ):
        """Set the key column for each feature in a list of uid and key pairs.

        Args:
            model: The model to update.
            key_column: The key column.
            uid_and_key: ``(uid, key)`` tuples.
            mc: The model context.
        """

        with model.db.connect() as conn:
            for uid, key in uid_and_key:
                sql = sa.update(
                    model.table()
                ).values({
                    key_column: key
                }).where(
                    model.uid_column().__eq__(uid)
                )
                conn.execute(sql)

    def drop_uid_and_key(
            self,
            model: gws.DatabaseModel,
            key_column: sa.Column,
            uids: Iterable[gws.FeatureUid],
            delete: bool,
            mc: gws.ModelContext,
    ):
        """Delete the features with the given uids, or set their key column to NULL.

        Args:
            model: The model to update.
            key_column: The key column.
            uids: The feature uids. None values are ignored.
            delete: If True, delete the features, otherwise clear their key.
            mc: The model context.
        """

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

        with model.db.connect() as conn:
            conn.execute(sql)

    def get_related(
            self,
            model: gws.DatabaseModel,
            uids: Iterable[gws.FeatureUid],
            mc: gws.ModelContext
    ) -> list[gws.Feature]:
        """Load related features by uid.

        Args:
            model: The related model.
            uids: The feature uids.
            mc: The model context, the features are loaded in a secondary context.

        Returns:
            The features.
        """
        return model.get_features(uids, gws.base.model.secondary_context(mc))

    def column_or_uid(self, model, cfg):
        """Return a column of a model, or its uid column.

        Args:
            model: The database model.
            cfg: The column name, or an empty value for the uid column.

        Returns:
            The column.
        """
        return model.column(cfg) if cfg else model.uid_column()

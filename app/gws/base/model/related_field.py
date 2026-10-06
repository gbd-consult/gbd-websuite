"""Base class for related fields."""

from typing import Optional, cast

import gws
import gws.base.model
import gws.lib.sa as sa

from . import field


class Config(field.Config):
    pass


class Props(field.Props):
    pass


class Link(gws.Data):
    """Link table of a many-to-many relationship."""

    tableName: str
    """The link table name."""
    srcKeyName: str
    """Column in the link table that refers to the source model."""
    dstKeyName: str
    """Column in the link table that refers to the related model."""


class RelRef(gws.Data):
    """One side of a relationship."""

    model: gws.DatabaseModel
    """The model."""
    keyName: str
    """The column that takes part in the relationship."""


class Relationship(gws.Data):
    """Relationship between the model of a field and related models."""

    src: RelRef
    """The side of the field's own model."""
    dst: RelRef
    """The related side, for relationships with a single related model."""
    dstList: list[RelRef]
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

    def post_configure(self):
        self.configure_relationship()

    def configure_relationship(self):
        """Set up the relationship ``rel``.

        Called after configuration. The base implementation does nothing.
        """
        pass

    def key_column(self, ref: RelRef) -> sa.Column:
        """Return the key column of a relationship side.

        Args:
            ref: The relationship side.

        Returns:
            The column ``ref.keyName`` of ``ref.model``.
        """
        return ref.model.column(ref.keyName)

    def link_table(self) -> sa.Table:
        """Return the link table of the relationship."""
        return self.model.db.table(self.rel.link.tableName)

    def link_column(self, name: str) -> sa.Column:
        """Return a column of the link table.

        Args:
            name: The column name.

        Returns:
            The column.
        """
        return self.model.db.column(self.link_table(), name)

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
        return [f for dst in self.rel.dstList for f in dst.model.find_features(search, mc)]

    def related_field(self, dst: RelRef) -> Optional[gws.ModelField]:
        """Find the field of a related model that describes the reverse relationship.

        Args:
            dst: The related side of the relationship.

        Returns:
            The field of ``dst.model`` whose relationship starts at ``dst.model`` and ``dst.keyName``,
            or None if there is no such field.
        """
        for fld in dst.model.fields:
            rel2 = cast(Relationship, getattr(fld, 'rel', None))
            if not rel2:
                continue
            if rel2.src.model == dst.model and rel2.src.keyName == dst.keyName:
                return fld

    def do_init_related(self, dst_feature, mc):
        our_features = [f for f in dst_feature.createWithFeatures if f.model == self.model]
        if not our_features:
            return

        for dst in self.rel.dstList:
            if dst.model == dst_feature.model:
                fld = self.related_field(dst)
                if fld:
                    if fld.attributeType == gws.AttributeType.feature:
                        dst_feature.attributes[fld.name] = our_features[0]
                    if fld.attributeType == gws.AttributeType.featurelist:
                        dst_feature.attributes.setdefault(fld.name, []).extend(our_features)

    def related_models(self):
        return [dst.model for dst in self.rel.dstList]

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
        dst_model_map: dict[str, gws.Model] = {dst.model.uid: dst.model for dst in self.rel.dstList}

        for v in value:
            rel_props = cast(gws.FeatureProps, gws.u.to_data_object(v))
            if rel_props:
                dst_model = dst_model_map.get(rel_props.modelUid)
                if dst_model:
                    related = dst_model.feature_from_props(rel_props, mc2)
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
        mc: gws.ModelContext,
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
        sql = sa.select(key_column).where(model.uid_equals(uid))
        with model.db.begin() as conn:
            rs = list(conn.execute(sql))
        return rs[0][0] if rs else None

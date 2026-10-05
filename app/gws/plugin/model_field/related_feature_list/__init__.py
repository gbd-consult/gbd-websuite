"""Related feature list field.

Represents a parent->child 1:M relationship to another model::

    +--------------+       +-------------+
    | parent table |       | child table |
    +--------------+       +-------------+
    | fromColumn   |-----<<| toColumn    |
    +--------------+       +-------------+

The value of the field is a list of child features. ``fromColumn`` is the key
column in this model's table, by default its primary key; ``toColumn`` is the
foreign key column in the child model.

The field is implemented as a ``relatedMultiFeatureList`` with a single child
model, so it reads and writes the relationship in the same way. Without a
configured widget, the field uses a ``featureList`` widget.

Example::

    fields+ {
        name "pois"
        type "relatedFeatureList"
        fromColumn "id"
        toModel "model_poi"
        toColumn "category_id"
        widget.type "featureList"
    }
"""

import gws
import gws.base.database.model
import gws.base.model.related_field as related_field

from gws.plugin.model_field import related_multi_feature_list


@gws.ext.config.modelField('relatedFeatureList')
class Config(related_field.Config):
    """Field listing related features in another model."""

    fromColumn: str = ''
    """Key column in this table, primary key by default."""
    toModel: str
    """UID of the related model."""
    toColumn: str
    """Foreign key column in the related model."""


@gws.ext.props.modelField('relatedFeatureList')
class Props(related_field.Props):
    pass


@gws.ext.object.modelField('relatedFeatureList')
class Object(related_multi_feature_list.Object):
    """Related feature list field object."""

    def configure_relationship(self):
        dst_mod = self.get_model(self.cfg('toModel'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                keyName=self.model.column(self.cfg('fromColumn') or self.model.uidName).name,
            ),
            dstList=[
                related_field.RelRef(
                    model=dst_mod,
                    keyName=dst_mod.column(self.cfg('toColumn')).name,
                )
            ],
        )
        self.rel.dst = self.rel.dstList[0]

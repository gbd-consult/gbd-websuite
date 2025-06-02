"""Related Feature List field

Represents a parent->child 1:M relationship to another model::

    +--------------+       +-------------+
    | parent table |       | child table |
    +--------------+       +-------------+
    | key          |-----<<| parent_key  |
    +--------------+       +-------------+

The value of the field is a list of child features.

This object is implemented as a "related_multi_feature_list" with a single target model.
"""

import gws
import gws.base.database.model
import gws.base.model.related_field as related_field

from gws.plugin.model_field import related_multi_feature_list

gws.ext.new.modelField('relatedFeatureList')


class Config(related_field.Config):
    """Configuration for related feature list field."""

    fromColumn: str = ''
    """Key column in this table, primary key by default."""
    toModel: str
    """Related model."""
    toColumn: str
    """Foreign key column in the related model."""


class Props(related_field.Props):
    pass


class Object(related_multi_feature_list.Object):
    def configure_relationship(self):
        to_mod = self.get_model(self.cfg('toModel'))

        self.rel = related_field.Relationship(
            src=related_field.RelRef(
                model=self.model,
                table=self.model.table(),
                key=self.column_or_uid(self.model, self.cfg('fromColumn')),
                uid=self.model.uid_column(),
            ),
            tos=[
                related_field.RelRef(
                    model=to_mod,
                    table=to_mod.table(),
                    key=to_mod.column(self.cfg('toColumn')),
                    uid=to_mod.uid_column(),
                )
            ],
        )
        self.rel.to = self.rel.tos[0]

import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedFeature')


# @TODO support composite keys
# @TODO support non-primary foreign keys

class RelationshipConfig(gws.Data):
    modelUid: str
    foreignKey: str


class Config(gws.base.model.field.Config):
    relationship: RelationshipConfig


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.feature

    model: gws.base.database.model.Object
    relationship: gws.base.model.field.Relationship

    def configure(self):
        self.relationship = gws.base.model.field.Relationship(self.cfg('relationship'))

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureSelect')
            return True

    def props(self, user):
        return gws.merge(super().props(user), relationships=[
            gws.base.model.field.RelationshipProps(
                modelUid=self.relationship.modelUid,
                fieldName=self.relationship.fieldName,
            )
        ])

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return

        val = props.attributes.get(self.name)
        if val is None:
            return

        rel_model = self.related_model()
        if not user.can_read(rel_model):
            return

        rel_rec = rel_model.get_record(gws.FeatureProps(val).uid)
        rel_feature = rel_model.feature_from_record(rel_rec, user, relation_depth - 1, **kwargs)
        if rel_feature:
            feature.attributes[self.name] = rel_feature

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return

        rel_rec = getattr(record, self.name, None)
        if rel_rec is None:
            return

        rel_model = self.related_model()
        if not user.can_read(rel_model):
            return

        rel_feature = rel_model.feature_from_record(rel_rec, user, relation_depth - 1, **kwargs)
        if rel_feature:
            feature.attributes[self.name] = rel_feature

    def store_to_record(self, feature, record, user, **kwargs):
        ok, rel_feature = self._value_to_write(feature)
        if not ok:
            return
        rel_model = self.related_model()
        rel_rec = rel_model.get_record(rel_feature.uid())
        if rel_rec is not None:
            setattr(record, self.name, rel_rec)

    ##

    def augment_select(self, sel, user):
        relation_depth = sel.search.relationDepth or 0
        if relation_depth <= 0:
            return
        sel.saSelect = sel.saSelect.options(
            sa.orm.selectinload(
                getattr(
                    self.model.record_class(),
                    self.name)
            ))

    def columns(self):
        rel_model = self.related_model()
        rel_key = rel_model.primary_keys()[0]
        return [
            sa.Column(self.relationship.foreignKey, sa.ForeignKey(rel_key))
        ]

    def orm_properties(self):
        rel_model = self.related_model()
        rel_cls = rel_model.record_class()
        rel_key = rel_model.primary_keys()[0]

        our_cls = self.model.record_class()

        kwargs = {}
        # kwargs['primaryjoin'] = getattr(our_cls, self.relationship.foreignKey) == getattr(rel_cls, rel_key.name)
        kwargs['foreign_keys'] = getattr(our_cls, self.relationship.foreignKey)
        # if self.relationship.fieldName:
        #     kwargs['back_populates'] = self.relationship.fieldName

        return {
            self.name: sa.orm.relationship(rel_cls, **kwargs)
        }

    def related_model(self) -> gws.IDatabaseModel:
        return self.model.provider.mgr.model(self.relationship.modelUid)

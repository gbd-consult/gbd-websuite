import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedFeature')


class FieldRef(gws.Data):
    type: t.Optional[str]
    name: str


class Config(gws.base.model.field.Config):
    foreignKey: FieldRef
    relation: gws.base.model.field.Relation


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.feature

    model: gws.base.database.model.Object
    relation: gws.base.model.field.Relation
    foreignKey: FieldRef

    def configure(self):
        self.relation = self.cfg('relation')
        self.foreignKey = self.cfg('foreignKey')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureSelect')
            return True

    def props(self, user):
        return gws.merge(super().props(user), relations=[self.relation])

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is not None:
            rel_model = self.related_model()
            rel_rec = rel_model.get_record(gws.FeatureProps(val).uid)
            rel_feature = rel_model.feature_from_record(rel_rec, user, relation_depth - 1, **kwargs)
            if rel_feature:
                feature.attributes[self.name] = rel_feature

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        if hasattr(record, self.name):
            rel_model = self.related_model()
            rel_rec = getattr(record, self.name)
            rel_feature = rel_model.feature_from_record(rel_rec, user, relation_depth - 1, **kwargs)
            if rel_feature:
                feature.attributes[self.name] = rel_feature

    def store_to_record(self, feature, record, user, **kwargs):
        ok, rel_feature = self._value_to_write(feature)
        if ok:
            rel_model = self.related_model()
            rel_rec = rel_model.get_record(rel_feature.uid())
            if rel_rec is not None:
                setattr(record, self.name, rel_rec)

    ##

    def augment_select(self, sel, user):
        depth = sel.search.relationDepth or 0
        if depth > 0:
            sel.saSelect = sel.saSelect.options(
                sa.orm.selectinload(
                    getattr(
                        self.model.record_class(),
                        self.name)
                ))

    def columns(self):
        rel_model = self.related_model()
        rel_keys = rel_model.primary_keys()
        return [
            sa.Column(self.foreignKey.name, sa.ForeignKey(rel_keys[0]))
        ]

    def orm_properties(self):
        rel_model = self.related_model()
        rel_cls = rel_model.record_class()
        kwargs = {}

        rel_pk = rel_model.primary_keys()[0]
        own_cls = self.model.record_class()

        kwargs['primaryjoin'] = getattr(own_cls, self.foreignKey.name) == getattr(rel_cls, rel_pk.name)

        if self.relation.fieldName:
            kwargs['back_populates'] = self.relation.fieldName

        return {
            self.name: sa.orm.relationship(rel_cls, **kwargs)
        }

    def related_model(self) -> gws.IDatabaseModel:
        return self.model.provider.mgr.model(self.relation.modelUid)

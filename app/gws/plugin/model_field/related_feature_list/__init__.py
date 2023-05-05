import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedFeatureList')


class Config(gws.base.model.field.Config):
    relation: gws.base.model.field.Relation


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.featurelist

    model: gws.base.database.model.Object
    relation: gws.base.model.field.Relation

    def configure(self):
        self.relation = self.cfg('relation')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureList')
            return True

    def props(self, user):
        return gws.merge(super().props(user), relations=[self.relation])

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is None:
            return
        rel_model = self.related_model()
        uids = [gws.FeatureProps(v).uid for v in val]
        recs = rel_model.get_records(uids)
        feature.attributes[self.name] = [
            rel_model.feature_from_record(rec, user, relation_depth - 1)
            for rec in recs
        ]

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        recs = getattr(record, self.name, None)
        if recs is None:
            return
        rel_model = self.related_model()
        feature.attributes[self.name] = [
            rel_model.feature_from_record(rec, user, relation_depth - 1)
            for rec in recs
        ]

    def store_to_record(self, feature, record, user, **kwargs):
        ok, feature_list = self._value_to_write(feature)
        if ok:
            rel_model = self.related_model()
            uids = [feature.uid() for feature in feature_list]
            recs = rel_model.get_records(uids)
            setattr(record, self.name, recs)

    def orm_properties(self):
        own_pk = self.model.primary_keys()[0]
        own_cls = self.model.record_class()

        rel_model = self.related_model()
        rel_cls = rel_model.record_class()
        rel_field_name = self.relation.fieldName
        rel_fk = getattr(rel_model.field(rel_field_name), 'foreignKey')

        kwargs = {}
        kwargs['primaryjoin'] = getattr(own_cls, own_pk.name) == getattr(rel_cls, rel_fk.name)
        kwargs['foreign_keys'] = getattr(rel_cls, rel_fk.name)
        kwargs['back_populates'] = rel_field_name

        if rel_model.sqlSort:
            order = []
            for s in rel_model.sqlSort:
                fn = sa.desc if s.reverse else sa.asc
                order.append(fn(getattr(rel_cls, s.fieldName)))
            kwargs['order_by'] = order

        return {
            self.name: sa.orm.relationship(rel_cls, **kwargs)
        }

    def related_model(self) -> gws.IDatabaseModel:
        return self.model.provider.mgr.model(self.relation.modelUid)

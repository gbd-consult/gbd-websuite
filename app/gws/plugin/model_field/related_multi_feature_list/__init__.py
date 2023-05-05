import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedMultiFeatureList')


class Config(gws.base.model.field.Config):
    relations: list[gws.base.model.field.Relation]


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.featurelist

    model: gws.base.database.model.Object
    relations: list[gws.base.model.field.Relation]

    def configure(self):
        self.relations = self.cfg('relations')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureList')
            return True

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return

        val = props.attributes.get(self.name)
        if val is None:
            return

        feature_list = []

        for v in val:
            fp = gws.FeatureProps(v)
            rel_model = self.model.provider.mgr.model(fp.modelUid)
            if not rel_model:
                continue
            # @TODO optimize
            rec = rel_model.get_record(fp.uid)
            if rec:
                feature_list.append(rel_model.feature_from_record(rec, user, relation_depth - 1))

        if feature_list:
            feature.attributes[self.name] = feature_list

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return

        feature_list = []

        for rel in self.relations:
            virt_key = gws.join_uid(self.name, rel.modelUid)
            recs = getattr(record, virt_key, None)
            if recs is None:
                continue
            rel_model = self.model.provider.mgr.model(rel.modelUid)
            feature_list.extend(
                rel_model.feature_from_record(rec, user, relation_depth - 1)
                for rec in recs
            )

        if feature_list:
            feature.attributes[self.name] = feature_list

    def store_to_record(self, feature, record, user, **kwargs):
        ok, val = self._value_to_write(feature)
        if ok:
            rel_model = self.related_model()
            uids = [f.uid() for f in val]
            recs = rel_model.get_records(uids)
            setattr(record, self.name, recs)

    ##

    def orm_properties(self):
        props = {}

        own_pk = self.model.primary_keys()[0]
        own_cls = self.model.record_class()

        for rel in self.relations:
            rel_model = self.model.provider.mgr.model(rel.modelUid)
            rel_cls = rel_model.record_class()
            rel_field_name = rel.fieldName
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

            virt_key = gws.join_uid(self.name, rel.modelUid)
            props[virt_key] = sa.orm.relationship(rel_cls, **kwargs)

        return props

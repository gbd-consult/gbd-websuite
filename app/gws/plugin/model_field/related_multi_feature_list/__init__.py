import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedMultiFeatureList')


class RelationshipConfig(gws.Config):
    modelUid: str
    fieldName: str


class Config(gws.base.model.field.Config):
    relationships: list[RelationshipConfig]


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.featurelist

    model: gws.base.database.model.Object
    relationships: list[gws.base.model.field.Relationship]

    def configure(self):
        self.relationships = self.cfg('relationships')

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureList')
            return True

    def props(self, user):
        return gws.merge(super().props(user), relationships=[
            gws.base.model.field.RelationshipProps(
                modelUid=rel.modelUid,
                fieldName=rel.fieldName,
            )
            for rel in self.relationships
        ])

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

        for rel in self.relationships:
            virt_col_name = gws.join_uid(self.name, rel.modelUid)
            recs = getattr(record, virt_col_name, None)
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
        ok, feature_list = self._value_to_write(feature)

        if not ok:
            return

        for rel in self.relationships:
            rel_model = self.model.provider.mgr.model(rel.modelUid)
            uids = [feature.uid() for feature in feature_list if feature.model.uid == rel_model.uid]
            recs = rel_model.get_records(uids)
            if recs:
                virt_col_name = gws.join_uid(self.name, rel.modelUid)
                setattr(record, virt_col_name, recs)

    ##

    def orm_properties(self):
        props = {}

        own_pk = self.model.primary_keys()[0]
        own_cls = self.model.record_class()

        for rel in self.relationships:
            rel_model = self.model.provider.mgr.model(rel.modelUid)
            rel_cls = rel_model.record_class()
            rel_field_name = rel.fieldName
            # rel_fk = getattr(rel_model.field(rel_field_name), 'foreignKey')

            kwargs = {}
            # kwargs['primaryjoin'] = getattr(own_cls, own_pk.name) == getattr(rel_cls, rel_fk.name)
            # kwargs['foreign_keys'] = getattr(rel_cls, rel_fk.name)
            kwargs['back_populates'] = rel_field_name

            if rel_model.sqlSort:
                order = []
                for s in rel_model.sqlSort:
                    fn = sa.desc if s.reverse else sa.asc
                    order.append(fn(getattr(rel_cls, s.fieldName)))
                kwargs['order_by'] = order

            virt_col_name = gws.join_uid(self.name, rel.modelUid)
            props[virt_col_name] = sa.orm.relationship(rel_cls, **kwargs)

        return props

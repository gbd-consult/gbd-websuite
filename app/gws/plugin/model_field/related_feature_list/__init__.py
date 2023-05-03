import gws
import gws.base.model.field
import gws.base.database.model
import gws.lib.sa as sa

import gws.types as t

gws.ext.new.modelField('relatedFeatureList')


class FieldRef(gws.Data):
    type: t.Optional[str]
    name: str


class Config(gws.base.model.field.Config):
    relation: gws.base.model.field.Relation


class Props(gws.base.model.field.Props):
    pass


class Object(gws.base.model.field.Object):
    attributeType = gws.AttributeType.featurelist

    model: gws.base.database.model.Object
    relations: list[gws.base.model.field.Relation]
    foreignKey: FieldRef

    def configure(self):
        self.relations = [self.cfg('relation')]

    def configure_widget(self):
        if not super().configure_widget():
            self.widget = self.create_child(gws.ext.object.modelWidget, type='featureList')
            return True
    ##

    def props(self, user):
        return gws.merge(super().props(user), relations=self.relations)

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

    ##

    def load_from_props(self, feature, props, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        val = props.attributes.get(self.name)
        if val is not None:
            rel_model = self.model.provider.mgr.model(self.relations[0].modelUid)
            uids = [gws.FeatureProps(v).attributes.get(rel_model.primary_keys()[0].name) for v in val]
            recs = rel_model.get_records(uids)
            feature.attributes[self.name] = [rel_model.feature_from_record(r, user, relation_depth - 1) for r in recs]

    def load_from_record(self, feature, record, user, relation_depth=0, **kwargs):
        if relation_depth <= 0:
            return
        if hasattr(record, self.name):
            val = getattr(record, self.name)
            rel_model = self.model.provider.mgr.model(self.relations[0].modelUid)
            recs = val
            feature.attributes[self.name] = [rel_model.feature_from_record(r, user, relation_depth - 1) for r in recs]

    def store_to_record(self, feature, record, user, **kwargs):
        ok, val = self._value_to_write(feature)
        if ok:
            rel_model = self.model.provider.mgr.model(self.relations[0].modelUid)
            uids = [f.uid() for f in val]
            recs = rel_model.get_records(uids)
            setattr(record, self.name, recs)

    ##

    def orm_properties(self):
        rel_model = t.cast(gws.IDatabaseModel, self.model.provider.mgr.model(self.relations[0].modelUid))
        rel_cls = rel_model.record_class()
        kwargs = {}

        own_pk = self.model.primary_keys()
        rel_fk = getattr(rel_model.field(self.relations[0].fieldName), 'foreignKey')
        own_cls = self.model.record_class()

        kwargs['primaryjoin'] = getattr(own_cls, own_pk[0].name) == getattr(rel_cls, rel_fk.name)
        kwargs['foreign_keys'] = getattr(rel_cls, rel_fk.name)
        kwargs['back_populates'] = self.relations[0].fieldName

        if rel_model.sqlSort:
            order = []
            for s in rel_model.sqlSort:
                fn = sa.desc if s.reverse else sa.asc
                order.append(fn(getattr(rel_cls, s.fieldName)))
            kwargs['order_by'] = order

        return {
            self.name: sa.orm.relationship(rel_cls, **kwargs)
        }

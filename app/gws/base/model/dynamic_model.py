import gws
import gws.base.feature
import gws.base.shape
import gws.types as t
from . import core

gws.ext.new.model('default')


class Config(core.Config):
    pass


class Object(core.Object, gws.IModel):
    def configure(self):
        self.uidName = 'uid'
        self.geometryName = 'geometry'
        self.loadingStrategy = gws.FeatureLoadingStrategy.all

        self.configure_fields()

    def feature_from_props(self, props, mc):
        feature = super().feature_from_props(props, mc)
        feature.attributes = dict(props.attributes)

        if self.geometryName:
            p = props.attributes.get(self.geometryName)
            if p:
                feature.attributes[self.geometryName] = gws.base.shape.from_props(p)

        return feature

    def feature_to_props(self, feature: gws.IFeature, mc):
        props = super().feature_to_props(feature, mc)
        props.attributes = dict(feature.attributes)
        if feature.shape():
            props.attributes[self.geometryName] = feature.shape().to_props()
        return props

    def feature_from_record(self, record: gws.FeatureRecord, mc: gws.ModelContext) -> t.Optional[gws.IFeature]:
        record = t.cast(gws.FeatureRecord, gws.to_data(record))
        feature = gws.base.feature.new(model=self, record=record)
        feature.attributes = record.attributes
        if record.uid and self.uidName:
            feature.attributes[self.uidName] = record.uid
        if record.shape and self.geometryName:
            feature.attributes[self.geometryName] = record.shape
        if record.meta:
            feature.category = record.meta.get('layerName', '')
        return feature

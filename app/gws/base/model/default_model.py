from typing import Optional, cast

import gws
import gws.base.feature
import gws.base.shape
from . import core

gws.ext.new.model('default')


class Config(core.Config):
    """Configuration for the default model."""

    pass


class Object(core.Object):
    def configure(self):
        self.uidName = core.DEFAULT_UID_NAME
        self.geometryName = core.DEFAULT_GEOMETRY_NAME
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

    def feature_to_props(self, feature: gws.Feature, mc):
        props = super().feature_to_props(feature, mc)
        props.attributes = dict(feature.attributes)
        s = feature.shape()
        if s is not None:
            props.attributes[self.geometryName] = s.to_props()
        return props

    def feature_from_record(self, record: gws.FeatureRecord, mc: gws.ModelContext) -> gws.Feature:
        record = cast(gws.FeatureRecord, gws.u.to_data_object(record))
        feature = gws.base.feature.new(model=self, record=record)
        feature.attributes = record.attributes
        if record.uid and self.uidName:
            feature.attributes[self.uidName] = record.uid
        if record.shape and self.geometryName:
            feature.attributes[self.geometryName] = record.shape
        if record.meta:
            feature.category = record.meta.get('layerName', '')
        return feature

"""Base model."""

import gws
import gws.base.feature
import gws.base.shape
import gws.types as t

from . import util


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: t.Optional[list[gws.ext.config.modelField]]
    """model fields"""
    loadingStrategy: t.Optional[gws.FeatureLoadingStrategy]
    """loading strategy for features"""
    templates: t.Optional[list[gws.ext.config.template]]
    """templates for this model"""
    title: str = ''
    """model title"""
    withAutoload: bool = False
    """autoload non-configured model fields from the source"""
    autoloadExclude: t.Optional[list[str]]
    """exclude field names from autoload"""


class Props(gws.Props):
    canCreate: bool
    canDelete: bool
    canRead: bool
    canWrite: bool
    fields: list[gws.ext.props.modelField]
    geometryCrs: t.Optional[str]
    geometryName: t.Optional[str]
    geometryType: t.Optional[gws.GeometryType]
    keyName: t.Optional[str]
    layerUid: t.Optional[str]
    loadingStrategy: gws.FeatureLoadingStrategy
    supportsGeometrySearch: bool
    supportsKeywordSearch: bool
    title: str
    uid: str


class Object(gws.Node, gws.IModel):
    def configure(self):
        self.fields = []
        self.geometryCrs = None
        self.geometryName = ''
        self.geometryType = None
        self.keyName = ''
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.templates = []
        self.title = self.cfg('title')

    def configure_fields(self):
        has_conf = False
        has_auto = False
        p = self.cfg('fields')
        if p:
            self.fields = self.create_children(gws.ext.object.modelField, p, _defaultModel=self)
            has_conf = True
        if not p or self.cfg('withAutoload'):
            has_auto = self.configure_auto_fields()
        return has_conf or has_auto

    def configure_auto_fields(self):
        exclude = set(self.cfg('autoloadExclude', default=[]))

        desc = self.describe()
        if not desc:
            return False

        for col in desc.columns.values():
            if col.isForeignKey:
                # we do not configure relations automatically
                # treating them as scalars leads to conflicts in sa Table classes
                continue
            cfg = util.field_config_from_column(col)
            if not cfg:
                continue
            name = cfg.get('name')
            if name in exclude or any(f.name == name for f in self.fields):
                continue
            fld = self.create_child(gws.ext.object.modelField, cfg, _defaultModel=self)
            if fld:
                self.fields.append(fld)

        return True

    def configure_templates(self):
        p = self.cfg('templates')
        if p:
            self.templates = gws.compact(self.configure_template(cfg) for cfg in p)
            return True

    def configure_template(self, cfg):
        return self.create_child(gws.ext.object.template, cfg)

    ##

    def props(self, user):
        layer = t.cast(gws.ILayer, self.closest(gws.ext.object.layer))
        return gws.Props(
            canCreate=user.can_create(self),
            canDelete=user.can_delete(self),
            canRead=user.can_read(self),
            canWrite=user.can_write(self),
            fields=self.fields,
            geometryCrs=self.geometryCrs.epsg if self.geometryCrs else None,
            geometryName=self.geometryName,
            geometryType=self.geometryType,
            keyName=self.keyName,
            layerUid=layer.uid if layer else None,
            loadingStrategy=self.loadingStrategy or (layer.loadingStrategy if layer else gws.FeatureLoadingStrategy.all),
            supportsGeometrySearch=any(f.supportsGeometrySearch for f in self.fields),
            supportsKeywordSearch=any(f.supportsKeywordSearch for f in self.fields),
            title=self.title or (layer.title if layer else ''),
            uid=self.uid,
        )

    ##

    def feature_from_data(self, data, user, relation_depth=0, **kwargs):
        feature = gws.base.feature.with_model(self)

        if self.fields:
            for f in self.fields:
                f.load_from_data(feature, data, user, relation_depth, **kwargs)
        else:
            feature.attributes = dict(data.attributes)
            if data.uid:
                feature.attributes[self.keyName] = data.uid
            if data.layerName:
                feature.layerName = data.layerName
            if data.shape:
                feature.attributes[self.geometryName] = data.shape

        return feature

    def feature_from_props(self, props, user, relation_depth=0, **kwargs):
        feature = gws.base.feature.with_model(self)

        if self.fields:
            for f in self.fields:
                f.load_from_props(feature, props, user, relation_depth, **kwargs)
        else:
            feature.attributes = dict(props.attributes)
            shape_props = props.attributes.get(self.geometryName)
            if shape_props:
                feature.attributes[self.geometryName] = gws.base.shape.from_props(shape_props)

        feature.cssSelector = gws.to_str(props.cssSelector)
        feature.views = gws.to_dict(props.views)
        feature.isNew = bool(props.isNew)
        return feature

    def feature_from_record(self, record, user, relation_depth=0, **kwargs):
        feature = gws.base.feature.with_model(self)

        for f in self.fields:
            f.load_from_record(feature, record, user, relation_depth, **kwargs)

        return feature

    def feature_props(self, feature, user, **kwargs):
        props = gws.FeatureProps(
            attributes={},
            cssSelector=feature.cssSelector,
            errors=feature.errors,
            geometryName=self.geometryName,
            isNew=feature.isNew,
            keyName=self.keyName,
            modelUid=self.uid,
            uid=feature.uid(),
            views=feature.views,
        )

        if self.fields:
            for f in self.fields:
                if user.can_read(f, self):
                    f.store_to_props(feature, props, user, **kwargs)
        else:
            props.attributes.update(feature.attributes)
            if feature.shape():
                props.attributes[self.geometryName] = gws.props(feature.shape(), user, self)

        return props

    ##

    def field(self, name):
        for f in self.fields:
            if f.name == name:
                return f

    def compute_values(self, feature, access, user, **kwargs):
        for f in self.fields:
            f.compute(feature, access, user, **kwargs)

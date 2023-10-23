"""Base model."""
from typing import Optional

import gws
import gws.base.feature
import gws.base.shape
import gws.config.util
import gws.types as t


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: t.Optional[list[gws.ext.config.modelField]]
    """model fields"""
    loadingStrategy: t.Optional[gws.FeatureLoadingStrategy]
    """loading strategy for features"""
    title: str = ''
    """model title"""
    isEditable: bool = False
    """this model is editable"""
    withAutoFields: bool = False
    """autoload non-configured model fields from the source"""
    excludeFields: t.Optional[list[str]]
    """exclude field names from autoload"""
    templates: t.Optional[list[gws.ext.config.template]]
    """feature templates"""


class Props(gws.Props):
    canCreate: bool
    canDelete: bool
    canRead: bool
    canWrite: bool
    isEditable: bool
    fields: list[gws.ext.props.modelField]
    geometryCrs: t.Optional[str]
    geometryName: t.Optional[str]
    geometryType: t.Optional[gws.GeometryType]
    layerUid: t.Optional[str]
    loadingStrategy: gws.FeatureLoadingStrategy
    supportsGeometrySearch: bool
    supportsKeywordSearch: bool
    title: str
    uid: str
    uidName: t.Optional[str]


class Object(gws.Node, gws.IModel):
    def configure(self):
        self.isEditable = self.cfg('isEditable')
        self.fields = []
        self.geometryCrs = None
        self.geometryName = ''
        self.geometryType = None
        self.uidName = ''
        self.loadingStrategy = self.cfg('loadingStrategy')
        self.title = self.cfg('title')

    def configure_fields(self):
        has_conf = False
        has_auto = False

        p = self.cfg('fields')
        if p:
            self.fields = self.create_children(gws.ext.object.modelField, p, _defaultModel=self)
            has_conf = True
        if not has_conf or self.cfg('withAutoFields'):
            has_auto = self.configure_auto_fields()

        return has_conf or has_auto

    def configure_auto_fields(self):
        desc = self.describe()
        if not desc:
            return False

        exclude = set(self.cfg('excludeFields', default=[]))
        exclude.update(fld.name for fld in self.fields)

        for col in desc.columns:
            if col.name in exclude:
                continue

            typ = _DEFAULT_FIELD_TYPES.get(col.type)
            if not typ:
                gws.log.warning(f'cannot find suitable field type for column {col.name!r} ({col.type})')
                continue

            cfg = gws.Config(
                type=typ,
                name=col.name,
                isPrimaryKey=col.isPrimaryKey,
                isRequired=not col.isNullable,
            )
            fld = self.create_child(gws.ext.object.modelField, cfg, _defaultModel=self)
            if fld:
                self.fields.append(fld)
                exclude.add(fld.name)

        return True

    def configure_uid(self):
        uids = []
        for fld in self.fields:
            if fld.isPrimaryKey:
                uids.append(fld.name)
        if len(uids) == 1:
            self.uidName = uids[0]
            return True

    def configure_geometry(self):
        for fld in self.fields:
            if getattr(fld, 'geometryType', None):
                self.geometryName = fld.name
                self.geometryType = getattr(fld, 'geometryType')
                self.geometryCrs = getattr(fld, 'geometryCrs')
                return True

    def configure_templates(self):
        return gws.config.util.configure_templates(self)

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
            isEditable=self.isEditable,
            layerUid=layer.uid if layer else None,
            loadingStrategy=self.loadingStrategy or (layer.loadingStrategy if layer else gws.FeatureLoadingStrategy.all),
            supportsGeometrySearch=any(fld.supportsGeometrySearch for fld in self.fields),
            supportsKeywordSearch=any(fld.supportsKeywordSearch for fld in self.fields),
            title=self.title or (layer.title if layer else ''),
            uid=self.uid,
            uidName=self.uidName,
        )

    ##

    def field(self, name):
        for fld in self.fields:
            if fld.name == name:
                return fld

    def validate_feature(self, feature, mc):
        feature.errors = []
        for fld in self.fields:
            fld.do_validate(feature, mc)
        return len(feature.errors) == 0

    def related_models(self):
        d = {}

        for fld in self.fields:
            for model in fld.related_models():
                d[model.uid] = model

        return list(d.values())

    ##

    def get_features(self, uids, mc):
        if not uids:
            return []
        search = gws.SearchQuery(uids=set(uids))
        return self.find_features(search, mc)

    def find_features(self, search, mc):
        return []

    ##

    def feature_from_props(self, props, mc):
        props = t.cast(gws.FeatureProps, gws.to_data(props))
        feature = gws.base.feature.with_model(self, props=props)
        for fld in self.fields:
            fld.from_props(feature, mc)
        return feature

    def feature_to_props(self, feature, mc):
        feature.props = gws.FeatureProps(
            attributes={},
            cssSelector=feature.cssSelector,
            errors=feature.errors or [],
            geometryName=self.geometryName,
            isNew=feature.isNew,
            uidName=self.uidName,
            modelUid=self.uid,
            uid=feature.uid(),
            views=feature.views,
        )

        for fld in self.fields:
            fld.to_props(feature, mc)

        return feature.props

    def feature_to_view_props(self, feature, mc):
        props = self.feature_to_props(feature, mc)

        a = {}

        if self.uidName:
            a[self.uidName] = props.attributes.get(self.uidName)
        if self.geometryName:
            a[self.geometryName] = props.attributes.get(self.geometryName)

        props.attributes = a
        props.modelUid = ''

        return props


##

# @TODO this should be populated dynamically from available gws.ext.object.modelField types

_DEFAULT_FIELD_TYPES = {
    gws.AttributeType.str: 'text',
    gws.AttributeType.int: 'integer',
    gws.AttributeType.date: 'date',
    gws.AttributeType.bool: 'bool',
    # gws.AttributeType.bytes: 'bytea',
    gws.AttributeType.datetime: 'datetime',
    # gws.AttributeType.feature: 'feature',
    # gws.AttributeType.featurelist: 'featurelist',
    gws.AttributeType.float: 'float',
    # gws.AttributeType.floatlist: 'floatlist',
    gws.AttributeType.geometry: 'geometry',
    # gws.AttributeType.intlist: 'intlist',
    # gws.AttributeType.strlist: 'strlist',
    gws.AttributeType.time: 'time',
}

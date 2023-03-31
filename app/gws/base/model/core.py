import gws
import gws.base.feature
import gws.types as t


class SortConfig:
    fieldName: str
    reverse: bool


class ValueConfig(gws.Config):
    pass


class Config(gws.ConfigWithAccess):
    """Model configuration"""

    fields: list[gws.ext.config.modelField]
    filter: t.Optional[str]
    sort: t.Optional[list[SortConfig]]
    loadingStrategy: t.Optional[gws.FeatureLoadingStrategy]
    """loading strategy for features"""


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
    uid: str


class Object(gws.Node, gws.IModel):
    def configure(self):
        self.fields = []
        self.keyName = ''
        self.geometryName = ''
        self.geometryType = None
        self.geometryCrs = None
        self.loadingStrategy = self.cfg('loadingStrategy')

    def configure_fields(self):
        p = self.cfg('fields')
        if p:
            for cfg in p:
                self.fields.append(
                    self.create_child(gws.ext.object.modelField, config=gws.merge(cfg, _model=self)))
            return True

    def props(self, user):
        layer = t.cast(gws.ILayer, self.parent)
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
            uid=self.uid,
        )

    def feature_from_data(self, data, user, **kwargs):
        feature = gws.base.feature.with_model(self)

        if self.fields:
            for f in self.fields:
                f.load_from_data(feature, data, user, **kwargs)
        else:
            feature.attributes = dict(data.attributes)
            if data.uid:
                feature.attributes[self.keyName] = data.uid
            if data.layerName:
                feature.layerName = data.layerName
            if data.shape:
                feature.attributes[self.geometryName] = data.shape

        return feature

    def feature_from_props(self, props, user, **kwargs):
        feature = gws.base.feature.with_model(self)

        if self.fields:
            for f in self.fields:
                f.load_from_props(feature, props, user, **kwargs)
        else:
            feature.attributes = dict(props.attributes)

        feature.isNew = bool(props.isNew)
        return feature

    def feature_from_record(self, record, user, **kwargs):
        feature = gws.base.feature.with_model(self)

        for f in self.fields:
            f.load_from_record(feature, record, user, **kwargs)

        return feature

    def feature_props(self, feature, user, **kwargs):
        props = gws.FeatureProps(
            attributes={},
            views=feature.views,
            uid=feature.uid(),
            isNew=feature.isNew,
            modelUid=self.uid,
            errors=feature.errors,
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

    def compute_values(self, feature, access, user, **kwargs):
        for f in self.fields:
            f.compute(feature, access, user, **kwargs)


##

def locate(
        models: list[gws.IModel],
        user: gws.IUser = None,
        access: gws.Access = None,
        uid: str = None
) -> t.Optional[gws.IModel]:
    for model in models:
        if user and access and not user.can(access, model):
            continue
        if uid and model.uid != uid:
            continue
        return model

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

    fields: t.List[gws.ext.config.modelField]
    filter: t.Optional[str]
    sort: t.Optional[t.List[SortConfig]]


class Props(gws.Props):
    geometryType: gws.GeometryType
    geometryName: str
    keyName: str
    layerUid: str
    crs: str
    fields: t.List[gws.ext.props.modelField]


class Object(gws.Node, gws.IModel):
    def configure(self):
        self.fields = []
        self.keyName = ''
        self.geometryName = ''

    def configure_fields(self):
        p = self.var('fields')
        if p:
            for cfg in p:
                self.fields.append(self.create_child(gws.ext.object.modelField, config=gws.merge(cfg, _model=self)))
            return True

    def feature_from_source(self, sf, user):
        atts = dict(sf.attributes)

        if sf.uid:
            atts[self.keyName] = sf.uid
        if sf.shape:
            atts[self.geometryName] = sf.shape
        if sf.layerName:
            atts['layerName'] = sf.layerName

        return self.feature_from_dict(atts, user)

    def feature_from_props(self, props, user):
        return self.feature_from_dict(t.cast(gws.base.feature.Props, props).attributes, user)

    def feature_from_dict(self, atts, user):
        feature = gws.base.feature.with_model(self)
        if not self.fields:
            feature.attributes = dict(atts)
            return feature
        for f in self.fields:
            f.load_from_dict(feature, atts, user)
        return feature

    def feature_from_record(self, record, user):
        feature = gws.base.feature.with_model(self)
        for f in self.fields:
            f.load_from_record(feature, record, user)
        return feature

    def feature_props(self, feature, user):
        if not self.fields:
            atts = feature.attributes
        else:
            atts = {}
            for f in self.fields:
                if user.can_read(f, self):
                    f.store_to_dict(feature, atts, user)

        uid = feature.uid()

        return gws.Props(
            attributes=atts,
            views=feature.views,
            uid=feature.uid(),
            isNew=feature.isNew,
            modelUid=self.uid,
        )

    def compute_values(self, feature, access, user, **kwargs):
        for f in self.fields:
            f.compute_value(feature, access, user, **kwargs)


##

def locate(
        models: t.List[gws.IModel],
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

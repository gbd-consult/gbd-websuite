import gws
import gws.gis.feature
import gws.gis.layer
import gws.gis.shape
import gws.types as t


class Config(gws.gis.layer.BaseConfig):
    """vector layer"""

    editStyle: t.Optional[t.StyleProps]  #: style for features being edited
    style: t.Optional[t.StyleProps]  #: style for features


class LayerProps(gws.gis.layer.BaseProps):
    style: t.Optional[t.StyleProps]
    editStyle: t.Optional[t.StyleProps]


class Object(gws.gis.layer.Base):
    style = None

    def configure(self):
        super().configure()
        self.style = self.var('style')

    @property
    def props(self):
        return gws.extend(super().props, {
            'style': self.style,
            'editStyle': self.var('editStyle'),
        })

    def get_features(self, bbox):
        shape = gws.gis.shape.from_bbox(bbox, self.crs)
        res = self.source.get_features(keyword=None, shape=shape)
        for f in res:
            f.normalize(self.meta)
            f.layer = self
        return res

    def modify_features(self, operation, feature_params):
        self.source.modify_features(operation, feature_params)

    def render_svg(self, bbox, dpi, scale, rotation, style):
        features = self.get_features(bbox)
        for f in features:
            f.set_default_style(style)
        return [f.to_svg(bbox, dpi, scale, rotation) for f in features]

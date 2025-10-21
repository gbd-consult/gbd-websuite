from typing import Optional

import gws
import gws.base.shape
import gws.lib.style
import gws.lib.svg


def new(
    model: gws.Model,
    attributes: Optional[dict] = None,
    record: Optional[gws.FeatureRecord] = None,
    props: Optional[gws.FeatureProps] = None,
) -> gws.Feature:
    f = Feature(model)
    f.attributes = attributes or {}
    f.record = record or gws.FeatureRecord(attributes={})
    f.props = props or gws.FeatureProps(attributes={})
    return f


class Feature(gws.Feature):
    def __init__(self, model: gws.Model):
        self.attributes = {}
        self.category = ''
        self.cssSelector = ''
        self.errors = []
        self.isNew = False
        self.model = model
        self.views = {}
        self.createWithFeatures = []
        self.insertedPrimaryKey = ''

    def __repr__(self):
        try:
            return f'<feature {self.model.uid}:{self.uid()}>'
        except Exception:
            return f'<feature ?>'

    def uid(self):
        if self.model.uidName:
            return str(self.attributes.get(self.model.uidName))
        return ''

    def shape(self):
        if self.model.geometryName:
            return self.attributes.get(self.model.geometryName)

    def get(self, name, default=None):
        return self.attributes.get(name, default)

    def has(self, name):
        return name in self.attributes

    def set(self, name, value):
        self.attributes[name] = value
        return self

    def raw(self, name):
        return self.record.attributes.get(name)

    def render_views(self, templates, **kwargs):
        tri = gws.TemplateRenderInput(
            args=gws.u.merge(
                self.attributes,
                kwargs,
                feature=self,
            )
        )
        for tpl in templates:
            view_name = tpl.subject.split('.')[-1]
            self.views[view_name] = tpl.render(tri).content
        return self

    def transform_to(self, crs) -> gws.Feature:
        s = self.shape()
        if s:
            self.attributes[self.model.geometryName] = s.transformed_to(crs)
        return self

    def to_svg(self, view, label=None, style=None):
        s = self.shape()
        if not s:
            return []
        shape = s.transformed_to(view.bounds.crs)
        return gws.lib.svg.shape_to_fragment(shape, view, label, style)

    def to_geojson(self, keep_crs=False) -> dict:
        d = {
            'type': 'Feature',
            'properties': {
                k: v  # noqa
                for k, v in self.attributes.items()
                if k != self.model.geometryName
            },
        }
        d['properties']['id'] = self.uid()
        s = self.shape()
        if s:
            d['geometry'] = s.to_geojson(keep_crs=keep_crs)
        return d

import os
import gws
import gws.common.model
import gws.types as t


class Object(gws.Object, t.TemplateObject):
    def configure(self):
        super().configure()
        self.data_model = self.add_child('gws.common.model', self.var('dataModel'))

    def dpi_for_quality(self, quality):
        q = self.var('qualityLevels')
        if q and quality < len(q):
            return q[quality].dpi
        return 0

    @property
    def props(self):
        return t.TemplateProps({
            'uid': self.uid,
            'title': self.var('title'),
            'qualityLevels': self.var('qualityLevels', default=[]),
            'dataModel': self.data_model,
            'mapWidth': self.map_size[0],
            'mapHeight': self.map_size[1],
            'pageWidth': self.page_size[0],
            'pageHeight': self.page_size[1],
        })

    def normalize_user_data(self, data: dict) -> dict:
        if not data or not self.data_model:
            return {}
        return self.data_model.apply(data)


# @TODO template types should be configurable

_types = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
}


def type_from_path(path):
    for ext, tt in _types.items():
        if path.endswith(ext):
            return tt


def config_from_path(path):
    tt = type_from_path(path)
    if tt:
        return t.Config({
            'type': tt,
            'path': path
        })


def from_path(path, tree):
    cnf = config_from_path(path)
    if cnf:
        return tree.create_object('gws.ext.template', cnf)


def builtin_config(name):
    # @TODO: cache
    # @TODO: do not hardcode template type

    if name == 'feature_format':
        return t.FeatureFormatConfig({
            'description': builtin_config('feature_description'),
            'teaser': builtin_config('feature_teaser')
        })

    path = os.path.dirname(__file__) + '/builtin_templates/' + name + '.cx.html'
    return config_from_path(path)

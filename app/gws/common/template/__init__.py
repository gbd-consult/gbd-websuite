import os
import gws.common.model

import gws.types as t


#:export
class TemplateQualityLevel(t.Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class Config(t.Config):
    type: str  #: template type
    qualityLevels: t.Optional[t.List[t.TemplateQualityLevel]]  #: list of quality levels supported by the template
    dataModel: t.Optional[gws.common.model.Config]  #: user-editable template attributes
    path: t.Optional[t.FilePath]  #: path to a template file
    text: str = ''  #: template content
    title: str = ''  #: template title


#:export
class TemplateProps(t.Props):
    uid: str
    title: str
    qualityLevels: t.List[t.TemplateQualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: t.ModelProps


#:export
class TemplateOutput(t.Data):
    mime: str
    content: str
    path: str


class FeatureFormatConfig(t.Config):
    """Feature format"""

    description: t.Optional[t.ext.template.Config]  #: template for feature descriptions
    category: t.Optional[t.ext.template.Config]  #: feature category
    label: t.Optional[t.ext.template.Config]  #: feature label on the map
    teaser: t.Optional[t.ext.template.Config]  #: template for feature teasers (short descriptions)
    title: t.Optional[t.ext.template.Config]  #: feature title


class LayerFormatConfig(t.Config):
    """Layer format"""

    description: t.Optional[t.ext.template.Config]  #: template for the layer description


#:export ITemplate
class Object(gws.Object, t.ITemplate):
    map_size: t.Size
    page_size: t.Size

    def __init__(self):
        super().__init__()
        self.data_model: t.IModel = None
        self.path = ''
        self.text = ''

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

    def configure(self):
        super().configure()

        self.path = self.var('path')
        self.text = self.var('text')

        uid = self.var('uid') or (gws.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

        p = self.var('dataModel')
        if p:
            self.data_model = self.add_child('gws.common.model', p)

    def dpi_for_quality(self, quality):
        q = self.var('qualityLevels')
        if q and quality < len(q):
            return q[quality].dpi
        return 0

    def normalize_context(self, context: dict) -> dict:
        if not self.data_model:
            return context
        atts = self.data_model.apply_to_dict(context)
        return {a.name: a.value for a in atts}

    def render(self, context: dict, render_output: t.RenderOutput = None, out_path: str = None, format: str = None) -> t.TemplateOutput:
        pass

    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str:
        pass


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
        return FeatureFormatConfig({
            'description': builtin_config('feature_description'),
            'teaser': builtin_config('feature_teaser')
        })

    path = os.path.dirname(__file__) + '/builtin_templates/' + name + '.cx.html'
    return config_from_path(path)

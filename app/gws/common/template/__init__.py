import os
import gws.common.model
import gws.tools.mime

import gws.types as t


#:export
class TemplateQualityLevel(t.Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class Config(t.WithType):
    dataModel: t.Optional[gws.common.model.Config]  #: user-editable template attributes
    mimeTypes: t.Optional[t.List[str]]  #: mime types this template can generate
    path: t.Optional[t.FilePath]  #: path to a template file
    subject: str = ''  #: template purpose
    qualityLevels: t.Optional[t.List[t.TemplateQualityLevel]]  #: list of quality levels supported by the template
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


#:export
class TemplateLegendMode(t.Enum):
    html = 'html'
    image = 'image'


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
    legend_mode: t.Optional[t.TemplateLegendMode]
    legend_layer_uids: t.List[str]

    @property
    def props(self):
        return t.TemplateProps(
            uid=self.uid,
            title=self.title,
            qualityLevels=self.var('qualityLevels', default=[]),
            dataModel=self.data_model,
            mapWidth=self.map_size[0],
            mapHeight=self.map_size[1],
            pageWidth=self.page_size[0],
            pageHeight=self.page_size[1],
        )

    def configure(self):
        super().configure()

        self.path: str = self.var('path')
        self.text: str = self.var('text')
        self.title: str = self.var('title')

        if self.path and not self.root.application.developer_option('template.reparse'):
            self.root.application.monitor.add_path(self.path)

        uid = self.var('uid') or (gws.sha256(self.path) if self.path else self.klass.replace('.', '_'))
        self.set_uid(uid)

        p = self.var('dataModel')
        self.data_model: t.Optional[t.IModel] = self.create_child('gws.common.model', p) if p else None

        self.subject: str = self.var('subject', default='').lower()
        p = self.subject.split('.')
        self.category: str = p[0] if len(p) > 1 else ''
        self.key: str = p[-1]

        self.mime_types: t.List[str] = []
        for p in self.var('mimeTypes', default=[]):
            self.mime_types.append(gws.tools.mime.get(p))

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

    def render(self, context: dict, mro: t.MapRenderOutput = None, out_path: str = None, legends: dict = None, format: str = None) -> t.TemplateOutput:
        pass

    def add_headers_and_footers(self, context: dict, in_path: str, out_path: str, format: str) -> str:
        pass


# @TODO template types should be configurable

_types = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
    '.cx.xml': 'xml',
}


def _type_from_path(path):
    for ext, tt in _types.items():
        if path.endswith(ext):
            return tt


def from_path(root, path, shared=True):
    tt = _type_from_path(path)
    if not tt:
        return
    cfg = t.Config(type=tt, path=path)
    if shared:
        return root.create_shared_object('gws.ext.template', '_template_' + path, cfg)
    return root.create_object('gws.ext.template', cfg)


def _builtin_configs():
    base = os.path.dirname(__file__) + '/builtin_templates/'

    return [
        t.Config(
            type='html',
            path=base + '/layer_description.cx.html',
            subject='layer.description',
        ),
        t.Config(
            type='html',
            path=base + '/project_description.cx.html',
            subject='project.description',
        ),
        t.Config(
            type='html',
            path=base + '/feature_description.cx.html',
            subject='feature.description',
        ),
        t.Config(
            type='html',
            path=base + '/feature_teaser.cx.html',
            subject='feature.teaser',
        ),
    ]


_builtin_templates = []


def builtins(root, category=None):
    if not _builtin_templates:
        for cfg in _builtin_configs():
            _builtin_templates.append(root.create_shared_object('gws.ext.template', '_builtin_template_' + cfg.path, cfg))
    return [tpl for tpl in _builtin_templates if not category or tpl.subject.startswith(category + '.')]


def configure_list(root, configs: t.List[t.ext.template.Config]) -> t.List[t.ITemplate]:
    if not configs:
        return []
    return [
        t.cast(t.ITemplate, root.create_object('gws.ext.template', c))
        for c in configs
    ]


def find(templates: t.List[t.ITemplate], subject: str = None, category: str = None, required: bool = False) -> t.Optional[t.ITemplate]:
    for tpl in templates:
        if subject and tpl.subject == subject:
            return tpl
        if category and tpl.category == category:
            return tpl

    if required:
        raise gws.Error(f'template not found: {subject or category}')

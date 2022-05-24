import os
import gws.common.model
import gws.tools.mime
import gws.tools.intl
import gws.tools.date

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
    qualityLevels: t.Optional[t.List[t.TemplateQualityLevel]]  #: quality levels supported by the template
    text: str = ''  #: template content
    title: str = ''  #: template title


class TemplateProps(t.Props):
    uid: str
    title: str
    qualityLevels: t.List[t.TemplateQualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: gws.common.model.Props


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
        return TemplateProps(
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

    def prepare_context(self, context: dict) -> dict:
        ext = {
            'gws': {
                'version': gws.VERSION,
                'endpoint': gws.SERVER_ENDPOINT,
            }
        }

        locale_uid = context.get('localeUid')
        if locale_uid:
            ext['locale'] = gws.tools.intl.locale(locale_uid)
            ext['date'] = gws.tools.date.date_formatter(locale_uid)
            ext['time'] = gws.tools.date.time_formatter(locale_uid)

        return gws.extend(context, ext)

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
    return from_config(root, cfg, shared=shared)


def from_config(root, cfg, shared: bool = False, parent: t.IObject = None) -> t.ITemplate:
    if not shared:
        return t.cast(t.ITemplate, root.create_object('gws.ext.template', cfg, parent))

    uid = gws.get(cfg, 'uid') or gws.get(cfg, 'path') or gws.sha256(gws.get(cfg, 'text') or '')
    tpl = t.cast(t.ITemplate, root.create_shared_object('gws.ext.template', uid, cfg))
    if parent:
        parent.append_child(tpl)
    return tpl


_dir = os.path.dirname(__file__) + '/builtin_templates/'

BUILTINS = [
    t.Config(
        type='html',
        path=_dir + '/layer_description.cx.html',
        subject='layer.description',
    ),
    t.Config(
        type='html',
        path=_dir + '/project_description.cx.html',
        subject='project.description',
    ),
    t.Config(
        type='html',
        path=_dir + '/feature_description.cx.html',
        subject='feature.description',
    ),
    t.Config(
        type='html',
        path=_dir + '/feature_teaser.cx.html',
        subject='feature.teaser',
    ),
]


def bundle(
        target: t.IObject,
        configs: t.List[t.ext.template.Config],
        defaults: t.List[t.ext.template.Config] = None,
) -> t.List[t.ITemplate]:
    ts = []

    for cfg in (configs or []):
        ts.append(from_config(target.root, cfg, shared=False, parent=target))

    for cfg in (defaults or []):
        ts.append(from_config(target.root, cfg, shared=True, parent=target))

    return ts


def find(templates: t.List[t.ITemplate], subject: str = None, category: str = None, mime: str = None) -> t.Optional[t.ITemplate]:
    for tpl in templates:
        ok = (
                (not subject or subject == tpl.subject)
                and (not category or category == tpl.category)
                and (not mime or mime in tpl.mime_types))
        if ok:
            return tpl

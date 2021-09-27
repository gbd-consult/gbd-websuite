import os

import gws
import gws.types as t
import gws.base.model
import gws.lib.date
import gws.lib.intl
import gws.lib.mime


class QualityLevel(gws.Data):
    """Quality level for a template"""

    name: str = ''  #: level name
    dpi: int  #: dpi value


class Config(gws.Config):
    dataModel: t.Optional[gws.base.model.Config]  #: user-editable template attributes
    mimeTypes: t.Optional[t.List[str]]  #: mime types this template can generate
    path: t.Optional[gws.FilePath]  #: path to a template file
    subject: str = ''  #: template purpose
    qualityLevels: t.Optional[t.List[QualityLevel]]  #: quality levels supported by the template
    text: str = ''  #: template content
    title: str = ''  #: template title


class Props(gws.Props):
    uid: str
    title: str
    qualityLevels: t.List[QualityLevel]
    mapHeight: int
    mapWidth: int
    dataModel: gws.base.model.Props


class LegendMode(t.Enum):
    html = 'html'
    image = 'image'


class Object(gws.Object, gws.ITemplate):
    key: str
    path: str
    subject: str
    text: str
    title: str

    data_model: t.Optional[gws.IDataModel]

    legend_layer_uids: t.List[str]
    legend_mode: t.Optional['LegendMode']
    legend_use_all: bool

    map_size: gws.Size
    page_size: gws.Size
    margin: gws.Extent

    @property
    def props(self):
        return Props(
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

        self.path: str = self.var('path')
        self.text: str = self.var('text')
        self.title: str = self.var('title')

        uid = self.var('uid') or (gws.sha256(self.path) if self.path else self.class_name.replace('.', '_'))
        self.set_uid(uid)

        self.data_model = self.create_child_if_config('gws.base.model', self.var('dataModel'))

        self.subject = self.var('subject', default='').lower()
        p = self.subject.split('.')
        self.category = p[0] if len(p) > 1 else ''
        self.key = p[-1]

        self.mime_types = []
        for p in self.var('mimeTypes', default=[]):
            self.mime_types.append(gws.lib.mime.get(p))

    def dpi_for_quality(self, quality):
        q = self.var('qualityLevels')
        if q and quality < len(q):
            return q[quality].dpi
        return 0

    def prepare_context(self, context: dict) -> dict:
        ext: t.Dict[str, t.Any] = {
            'gws': {
                'version': gws.VERSION,
                'endpoint': gws.SERVER_ENDPOINT,
            }
        }

        locale_uid = context.get('localeUid')
        if locale_uid:
            ext['locale'] = gws.lib.intl.locale(locale_uid)
            ext['date'] = gws.lib.date.date_formatter(locale_uid)
            ext['time'] = gws.lib.date.time_formatter(locale_uid)

        return gws.merge(ext, context)

    def add_page_elements(self, context: dict, in_path: str, out_path: str, format: str):
        return in_path


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


def from_path(root: gws.RootObject, path, shared=True) -> t.Optional[Object]:
    tt = _type_from_path(path)
    if not tt:
        return None
    cfg = gws.Config(type=tt, path=path)
    return from_config(root, cfg, shared=shared)


def from_config(root: gws.RootObject, cfg: gws.Config, shared: bool = False, parent: gws.Object = None) -> t.Optional[Object]:
    if not shared:
        return t.cast(Object, root.create_object('gws.ext.template', cfg, parent))

    uid = gws.get(cfg, 'uid') or gws.get(cfg, 'path') or gws.sha256(gws.get(cfg, 'text') or '')
    return t.cast(Object, root.create_shared_object('gws.ext.template', uid, cfg))


_dir = os.path.dirname(__file__) + '/builtin_templates/'
_public = [{'role': 'all', 'type': 'allow'}]

BUILTINS = [
    gws.Config(
        type='html',
        path=_dir + '/layer_description.cx.html',
        subject='layer.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/project_description.cx.html',
        subject='project.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/feature_description.cx.html',
        subject='feature.description',
        access=_public,
    ),
    gws.Config(
        type='html',
        path=_dir + '/feature_teaser.cx.html',
        subject='feature.teaser',
        access=_public,
    ),
]


#


class BundleConfig(gws.Config):
    templates: t.List[Config]
    defaults: t.List[Config]


class BundleProps:
    items: t.List[Props]


class Bundle(gws.Object, gws.ITemplateBundle):
    items: t.List[gws.ITemplate]

    @property
    def props(self):
        return gws.Props(items=self.items)

    def configure(self):
        self.items = []

        p = self.var('templates')
        if p:
            for cfg in p:
                self.items.append(self._create_template(cfg))

        subjects = set(tpl.subject for tpl in self.items)

        p = self.var('defaults')
        if p:
            for cfg in p:
                if cfg.get('subject') not in subjects:
                    self.items.append(self._create_template(cfg))

    def _create_template(self, cfg):
        if cfg.get('access'):
            return from_config(self.root, cfg, shared=True)
        return from_config(self.root, cfg, shared=False, parent=self)

    def all(self) -> t.List[gws.ITemplate]:
        return self.items

    def find(self, subject: str = None, category: str = None, mime: str = None) -> t.Optional[gws.ITemplate]:
        for tpl in self.items:
            ok = (
                    (not subject or subject == tpl.subject)
                    and (not category or category == tpl.category)
                    and (not mime or mime in tpl.mime_types))
            if ok:
                return tpl


def create_bundle(
        parent: gws.Object,
        templates: t.List[Config],
        defaults: t.List[Config] = None,
        with_builtins: bool = False) -> Bundle:
    if with_builtins:
        defaults = BUILTINS

    return t.cast(
        Bundle,
        parent.create_child(
            Bundle,
            BundleConfig(templates=templates, defaults=defaults)))

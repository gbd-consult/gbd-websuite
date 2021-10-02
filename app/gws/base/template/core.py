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


##


# @TODO template types should be configurable

_types = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
    '.cx.xml': 'xml',
}


def create_from_path(root: gws.RootObject, path, shared: bool = False, parent: gws.Object = None) -> t.Optional['Object']:
    for ext, typ in _types.items():
        if path.endswith(ext):
            return create(root, gws.Config(type=typ, path=path), shared=shared, parent=parent)


def create(root: gws.RootObject, cfg: gws.Config, shared: bool = False, parent: gws.Object = None) -> 'Object':
    if not shared:
        return t.cast(Object, root.create_object('gws.ext.template', cfg, parent))
    uid = gws.get(cfg, 'uid') or gws.get(cfg, 'path') or gws.sha256(gws.get(cfg, 'text', default=''))
    return t.cast(Object, root.create_shared_object('gws.ext.template', cfg, uid))

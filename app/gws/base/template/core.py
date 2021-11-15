import os

import gws
import gws.base.model
import gws.lib.date
import gws.lib.intl
import gws.lib.mime
import gws.lib.os2
import gws.types as t


class Config(gws.Config):
    dataModel: t.Optional[gws.base.model.Config]  #: user-editable template attributes
    mapSize: t.Optional[gws.MSize]
    mimeTypes: t.Optional[t.List[str]]  #: mime types this template can generate
    pageSize: t.Optional[gws.MSize]
    path: t.Optional[gws.FilePath]  #: path to a template file
    qualityLevels: t.Optional[t.List[gws.TemplateQualityLevel]]  #: quality levels supported by the template
    subject: str = ''  #: template purpose
    text: str = ''  #: template content
    title: str = ''  #: template title


class Props(gws.Props):
    dataModel: gws.base.model.Props
    mapSize: t.Optional[gws.MSize]
    pageSize: t.Optional[gws.MSize]
    qualityLevels: t.List[gws.TemplateQualityLevel]
    title: str
    uid: str


class Object(gws.Node, gws.ITemplate):

    def props_for(self, user):
        return gws.Data(
            dataModel=self.data_model,
            mapSize=self.map_size,
            pageSize=self.page_size,
            qualityLevels=self.quality_levels,
            title=self.title,
            uid=self.uid,
        )

    def configure(self):
        self.path = self.var('path')
        self.text = self.var('text', default='')
        self.title = self.var('title', default='')

        uid = self.var('uid') or (gws.sha256(self.path) if self.path else self.class_name.replace('.', '_'))
        self.set_uid(uid)

        self.quality_levels = self.var('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        self.data_model = self.create_child_if_config('gws.base.model', self.var('dataModel'))

        self.subject = self.var('subject', default='').lower()
        if '.' in self.subject:
            self.category, _, self.name = self.subject.partition('.')
        else:
            self.category, self.name = '', self.subject

        self.mimes = []
        for p in self.var('mimeTypes', default=[]):
            self.mimes.append(gws.lib.mime.get(p))

        self.map_size = self.var('mapSize')
        self.page_size = self.var('pageSize')

    def prepare_context(self, context: dict) -> dict:
        ctx = context or {}
        ext = {
            'gws': {
                'version': gws.VERSION,
                'endpoint': gws.SERVER_ENDPOINT,
            }
        }

        locale_uid = ctx.get('localeUid')
        if locale_uid:
            ext['locale'] = gws.lib.intl.locale(locale_uid)
            ext['date'] = gws.lib.date.date_formatter(locale_uid)
            ext['time'] = gws.lib.date.time_formatter(locale_uid)

        return gws.merge(ext, ctx)


##


# @TODO template types should be configurable

_types = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
    '.cx.xml': 'xml',
}


def create_from_path(root: gws.IRoot, path, parent: gws.Node = None, shared: bool = False) -> t.Optional['Object']:
    for ext, typ in _types.items():
        if path.endswith(ext):
            return create(root, gws.Config(type=typ, path=path), parent, shared)


def create(root: gws.IRoot, cfg: gws.Config, parent: gws.Node = None, shared: bool = False) -> Object:
    key = gws.get(cfg, 'uid') or gws.get(cfg, 'path') or gws.sha256(gws.get(cfg, 'text', default=''))
    return root.create_object('gws.ext.template', cfg, parent, shared, key)

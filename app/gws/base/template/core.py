import os

import gws
import gws.base.model
import gws.lib.date
import gws.lib.intl
import gws.lib.mime
import gws.types as t


class Config(gws.Config):
    dataModel: t.Optional[gws.base.model.Config]
    """user-editable template attributes"""
    mapSize: t.Optional[gws.MSize]
    mimeTypes: t.Optional[t.List[str]]
    """mime types this template can generate"""
    pageSize: t.Optional[gws.MSize]
    path: t.Optional[gws.FilePath]
    """path to a template file"""
    qualityLevels: t.Optional[t.List[gws.TemplateQualityLevel]]
    """quality levels supported by the template"""
    subject: str = ''
    """template purpose"""
    text: str = ''
    """template content"""
    title: str = ''
    """template title"""


class Props(gws.Props):
    model: t.Optional[gws.base.model.Props]
    mapSize: t.Optional[gws.MSize]
    pageSize: t.Optional[gws.MSize]
    qualityLevels: t.List[gws.TemplateQualityLevel]
    title: str
    uid: str


class Object(gws.Node, gws.ITemplate):
    def configure(self):
        self.path = self.var('path')
        self.text = self.var('text', default='')
        self.title = self.var('title', default='')

        self.qualityLevels = self.var('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        # self.data_model = self.root.create_optional('gws.base.model', self.var('dataModel'))

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

    def props(self, user):
        return gws.Data(
            # dataModel=self.data_model,
            mapSize=self.map_size,
            pageSize=self.page_size,
            qualityLevels=self.qualityLevels,
            title=self.title,
            uid=self.uid,
        )

    def prepare_args(self, args: dict) -> dict:
        args = args or {}
        ext = {
            'gws': {
                'version': self.root.app.version,
                'endpoint': gws.SERVER_ENDPOINT,
            }
        }

        locale_uid = args.get('localeUid')
        if locale_uid:
            ext['locale'] = gws.lib.intl.locale(locale_uid)
            ext['date'] = gws.lib.date.date_formatter(locale_uid)
            ext['time'] = gws.lib.date.time_formatter(locale_uid)

        return gws.merge(ext, args)


##


def locate(
        templates: t.List[gws.ITemplate],
        user: gws.IUser = None,
        subject: str = None,
        mime: str = None
) -> t.Optional[gws.ITemplate]:
    mt = gws.lib.mime.get(mime) if mime else None

    for tpl in templates:
        if user and not user.can_use(tpl):
            continue
        if mt and tpl.mimes and mt not in tpl.mimes:
            continue
        if subject and tpl.subject != subject:
            continue
        return tpl


def render(
        templates: t.List[gws.ITemplate],
        tri: gws.TemplateRenderInput,
        user: gws.IUser = None,
        subject: str = None,
        mime: str = None
) -> t.Optional[gws.ContentResponse]:
    tpl = locate(templates, user, subject, mime)
    if not tpl:
        return
    return tpl.render(tri)


##


# @TODO template types should be configurable

_types = {
    '.cx.html': 'html',
    '.cx.csv': 'csv',
    '.qgs': 'qgis',
    '.cx.xml': 'xml',
}


def from_path(root: gws.IRoot, path) -> t.Optional['Object']:
    for ext, typ in _types.items():
        if path.endswith(ext):
            return root.create_shared(gws.ext.object.template, gws.Config(uid=gws.sha256(path), type=typ, path=path))

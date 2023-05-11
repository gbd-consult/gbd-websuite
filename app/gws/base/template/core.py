import os

import gws
import gws.base.model
import gws.lib.date
import gws.lib.intl
import gws.lib.mime
import gws.types as t


class Config(gws.Config):
    models: t.Optional[list[gws.ext.config.model]]
    """data models"""
    mapSize: t.Optional[gws.MSize]
    """map size"""
    mimeTypes: t.Optional[list[str]]
    """mime types this template can generate"""
    pageSize: t.Optional[gws.MSize]
    """page size"""
    pageMargin: t.Optional[gws.MExtent]
    """page margin"""
    path: t.Optional[gws.FilePath]
    """path to a template file"""
    qualityLevels: t.Optional[list[gws.TemplateQualityLevel]]
    """quality levels supported by this template"""
    subject: str = ''
    """template purpose"""
    text: str = ''
    """template content"""
    title: str = ''
    """template title"""


class Props(gws.Props):
    model: t.Optional[gws.base.model.Props]
    mapSize: t.Optional[gws.Size]
    pageSize: t.Optional[gws.Size]
    qualityLevels: list[gws.TemplateQualityLevel]
    title: str


DEFAULT_MAP_SIZE = (50, 50, gws.Uom.mm)
DEFAULT_PAGE_SIZE = (210, 297, gws.Uom.mm)


class Object(gws.Node, gws.ITemplate):
    path: str
    text: str
    title: str

    def configure(self):
        self.path = self.cfg('path')
        self.text = self.cfg('text', default='')

        if not self.path and not self.text:
            raise gws.Error('either "path" or "text" required')

        self.title = self.cfg('title', default='')

        self.qualityLevels = self.cfg('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        # self.data_model = self.root.create_optional('gws.base.model', self.cfg('dataModel'))

        self.subject = self.cfg('subject', default='').lower()

        self.mimes = []
        for p in self.cfg('mimeTypes', default=[]):
            self.mimes.append(gws.lib.mime.get(p))

        self.mapSize = self.cfg('mapSize') or DEFAULT_MAP_SIZE
        self.pageSize = self.cfg('pageSize') or DEFAULT_PAGE_SIZE
        self.pageMargin = self.cfg('pageMargin')

    def props(self, user):
        return gws.Data(
            # dataModel=self.data_model,
            mapSize=self.mapSize,
            pageSize=self.pageSize,
            qualityLevels=self.qualityLevels,
            title=self.title,
            uid=self.uid,
        )

    def prepare_args(self, args: dict) -> dict:
        args = args or {}
        locale_uid = args.get('localeUid', 'en_CA')

        extra = dict(
            gwsVersion=self.root.app.version,
            gwsBaseUrl=gws.SERVER_ENDPOINT,
            locale=gws.lib.intl.locale(locale_uid),
            date=gws.lib.date.date_formatter(locale_uid),
            time=gws.lib.date.time_formatter(locale_uid),
        )

        return gws.merge(extra, args)


##


def locate(
        *objects,
        user: gws.IUser = None,
        subject: str = None,
        mime: str = None
) -> t.Optional[gws.ITemplate]:
    mt = gws.lib.mime.get(mime) if mime else None

    for obj in objects:
        if not obj:
            continue
        for tpl in getattr(obj, 'templates', []):
            if user and not user.can_use(tpl):
                continue
            if mt and tpl.mimes and mt not in tpl.mimes:
                continue
            if subject and tpl.subject != subject:
                continue
            return tpl


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

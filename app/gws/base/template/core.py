import os

import gws
import gws.base.model
import gws.config.util
import gws.lib.intl
import gws.lib.mime
import gws.types as t


class Config(gws.Config):
    mapSize: t.Optional[gws.MSize]
    """map size"""
    mimeTypes: t.Optional[list[str]]
    """mime types this template can generate"""
    pageSize: t.Optional[gws.MSize]
    """page size"""
    pageMargin: t.Optional[gws.MExtent]
    """page margin"""
    qualityLevels: t.Optional[list[gws.TemplateQualityLevel]]
    """quality levels supported by this template"""
    subject: str = ''
    """template purpose"""
    title: str = ''
    """template title"""


class Props(gws.Props):
    mapSize: t.Optional[gws.Size]
    pageSize: t.Optional[gws.Size]
    qualityLevels: list[gws.TemplateQualityLevel]
    title: str


DEFAULT_MAP_SIZE = (50, 50, gws.Uom.mm)
DEFAULT_PAGE_SIZE = (210, 297, gws.Uom.mm)


class Object(gws.Node, gws.ITemplate):
    title: str

    def configure(self):
        self.title = self.cfg('title', default='')
        self.qualityLevels = self.cfg('qualityLevels') or [gws.TemplateQualityLevel(name='default', dpi=0)]
        self.subject = self.cfg('subject', default='')

        self.mimes = []
        for p in self.cfg('mimeTypes', default=[]):
            self.mimes.append(gws.lib.mime.get(p))

        self.mapSize = self.cfg('mapSize') or DEFAULT_MAP_SIZE
        self.pageSize = self.cfg('pageSize') or DEFAULT_PAGE_SIZE
        self.pageMargin = self.cfg('pageMargin')

    def props(self, user):
        return gws.Data(
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
            date=gws.lib.intl.date_formatter(locale_uid),
            time=gws.lib.intl.time_formatter(locale_uid),
        )

        return gws.merge(extra, args)

    def notify(self, tri: gws.TemplateRenderInput, message: str):
        if tri.notify:
            tri.notify(message)

from typing import Optional

import os

import gws
import gws.base.model
import gws.config.util
import gws.lib.intl
import gws.lib.mime


class Config(gws.Config):
    mapSize: Optional[gws.UomSizeStr]
    """map size"""
    mimeTypes: Optional[list[str]]
    """mime types this template can generate"""
    pageSize: Optional[gws.UomSizeStr]
    """page size"""
    pageMargin: Optional[gws.UomExtentStr]
    """page margin"""
    subject: str = ''
    """template purpose"""
    title: str = ''
    """template title"""


class Props(gws.Props):
    mapSize: Optional[gws.Size]
    pageSize: Optional[gws.Size]
    title: str


DEFAULT_MAP_SIZE = (50, 50, gws.Uom.mm)
DEFAULT_PAGE_SIZE = (210, 297, gws.Uom.mm)


class Object(gws.Template):
    def configure(self):
        self.title = self.cfg('title', default='')
        self.subject = self.cfg('subject', default='')

        self.mimeTypes = []
        for p in self.cfg('mimeTypes', default=[]):
            m = gws.lib.mime.get(p)
            if not m:
                raise gws.ConfigurationError(f'invalid mime type {p!r}')
            self.mimeTypes.append(m)

        self.mapSize = self.cfg('mapSize') or DEFAULT_MAP_SIZE
        self.pageSize = self.cfg('pageSize') or DEFAULT_PAGE_SIZE
        self.pageMargin = self.cfg('pageMargin')

    def props(self, user):
        return gws.Data(
            mapSize=self.mapSize,
            pageSize=self.pageSize,
            title=self.title,
            uid=self.uid,
        )

    def prepare_args(self, tri: gws.TemplateRenderInput):
        args = tri.args or {}
        args.setdefault('app', self.root.app)

        locale = args.get('locale') or tri.locale
        if not locale:
            ls = self.root.app.localeUids
            if ls:
                locale = gws.lib.intl.locale(ls[0], fallback=True)
        if not locale:
            locale =  gws.lib.intl.default_locale()

        f = gws.lib.intl.formatters(locale)

        args.setdefault('locale', locale)
        args.setdefault('date', f[0])
        args.setdefault('time', f[1])
        args.setdefault('number', f[2])

        # obsolete
        args.setdefault('gwsVersion', self.root.app.version)
        args.setdefault('gwsBaseUrl', gws.c.SERVER_ENDPOINT)

        return args

    def notify(self, tri: gws.TemplateRenderInput, message: str):
        if tri.notify:
            tri.notify(message)

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

    def prepare_args(self, args):
        args = args or {}
        locale_uid = args.get('localeUid', 'en_CA')
        args.setdefault('app', self.root.app)
        args.setdefault('locale', gws.lib.intl.locale(locale_uid))
        args.setdefault('date', gws.lib.intl.date_formatter(locale_uid))
        args.setdefault('time', gws.lib.intl.time_formatter(locale_uid))
        args.setdefault('number', gws.lib.intl.number_formatter(locale_uid))
        # obsolete
        args.setdefault('gwsVersion', self.root.app.version)
        args.setdefault('gwsBaseUrl', gws.c.SERVER_ENDPOINT)

        return args

    def notify(self, tri: gws.TemplateRenderInput, message: str):
        if tri.notify:
            tri.notify(message)

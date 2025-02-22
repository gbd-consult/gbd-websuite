from typing import Optional

import gws

from . import site

_FALLBACK_SITE = gws.Config(
    host='*',
    root=site.WebDocumentRootConfig(dir='/data/web'))


class Config(gws.Config):
    """Web server configuration"""

    sites: Optional[list[site.Config]]
    """configured sites"""
    ssl: Optional[site.SSLConfig]
    """ssl configuration"""


class Object(gws.WebManager):
    def configure(self):
        cfgs = self.cfg('sites', default=[])
        if all(c.host != '*' for c in cfgs):
            cfgs.append(_FALLBACK_SITE)
        if self.cfg('ssl'):
            cfgs = [gws.u.merge(c, ssl=True) for c in cfgs]
        self.sites = self.create_children(site.Object, cfgs)

        self.root.app.middlewareMgr.register(self, 'cors')

    ##

    def enter_middleware(self, req: gws.WebRequester):
        cors = req.site.corsOptions
        if not cors:
            return
        if req.method == 'OPTIONS':
            return gws.ContentResponse(mime='text/plain', content='')

    def exit_middleware(self, req: gws.WebRequester, res: gws.WebResponder):
        cors = req.site.corsOptions

        if not cors:
            return

        p = cors.allowOrigin
        if p:
            res.add_header('Access-Control-Allow-Origin', p)

        p = cors.allowCredentials
        if p:
            res.add_header('Access-Control-Allow-Credentials', 'true')

        p = cors.allowHeaders
        if p:
            res.add_header('Access-Control-Allow-Headers', p)

        p = cors.allowMethods
        if p:
            res.add_header('Access-Control-Allow-Methods', p)
        else:
            res.add_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

    ##

    def site_from_environ(self, environ):
        host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

        for s in self.sites:
            if s.host.lower() == host:
                return s
        for s in self.sites:
            if s.host == '*':
                return s

        # there must be a '*' site
        raise gws.Error('unknown host', host)

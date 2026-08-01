from typing import Optional

import gws

from . import site

class Config(gws.Config):
    """Web server configuration"""

    site: Optional[site.Config]
    """Site configuration. (added in 8.4)"""
    sites: Optional[list[site.Config]]
    """Sites configuration. (deprecated in 8.4)"""
    ssl: Optional[site.SSLConfig]
    """SSL configuration."""


class Object(gws.WebManager):
    def configure(self):
        p = self.cfg('site')
        if not p:
            # deprecated
            cfgs = self.cfg('sites') or []
            if len(cfgs) > 1:
                raise gws.ConfigurationError('multiple web sites are not supported')
            p = cfgs[0] if cfgs else gws.Config()
        if self.cfg('ssl'):
            p = gws.u.merge(p, ssl=True)
        self.site = self.create_child(site.Object, p)

        # deprecated
        self.sites = [self.site]
        
        self.root.app.middlewareMgr.register(self, 'cors')

    ##

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

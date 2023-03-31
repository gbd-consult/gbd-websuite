import gws
import gws.types as t

from . import site

_FALLBACK_SITE = gws.Config(
    host='*',
    root=site.DocumentRootConfig(dir='/data/web'))


class Config(gws.Config):
    """Web server configuration"""

    sites: t.Optional[list[site.Config]] 
    """configured sites"""
    ssl: t.Optional[site.SSLConfig] 
    """ssl configuration"""


class Object(gws.Node, gws.IWebManager):
    def configure(self):
        cfgs = self.cfg('sites', default=[])
        if all(c.host != '*' for c in cfgs):
            cfgs.append(_FALLBACK_SITE)
        if self.cfg('ssl'):
            cfgs = [gws.merge(c, ssl=True) for c in cfgs]
        self.sites = self.create_children(site.Object, cfgs)

    def activate(self):
        self.root.app.register_web_middleware('cors', self.cors_middleware)

    def cors_middleware(self, req: gws.IWebRequester, nxt):
        cors = req.site.corsOptions
        if not cors:
            return nxt()

        if req.method == 'OPTIONS':
            return gws.ContentResponse(mime='text/plain', content='')

        res = nxt()

        if res.status < 400:

            p = cors.get('allowOrigin')
            if p:
                res.add_header('Access-Control-Allow-Origin', p)

            p = cors.get('allowCredentials')
            if p:
                res.add_header('Access-Control-Allow-Credentials', 'true')

            p = cors.get('allowHeaders')
            if p:
                res.add_header('Access-Control-Allow-Headers', ', '.join(p))

            p = cors.get('allowMethods')
            if p:
                res.add_header('Access-Control-Allow-Methods', ', '.join(p))
            else:
                res.add_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

        return res

    def site_from_environ(self, environ):
        host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

        for s in self.sites:
            if s.host.lower() == host:
                return s
        for s in self.sites:
            if s.host == '*':
                return s

        # there must be a '*' site
        raise ValueError('unknown host', host)

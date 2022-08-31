import gws
import gws.types as t

from . import main


class Config(gws.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[main.Config]]  #: configured sites
    ssl: t.Optional[main.SSLConfig]  #: ssl configuration


class Object(gws.Node, gws.IWebSiteCollection):

    def configure(self):
        cfgs = self.var('sites', default=[])
        if all(c.host != '*' for c in cfgs):
            cfgs.append(gws.Config(host='*', root=main.DocumentRootConfig(dir='/data/web')))
        if self.var('ssl'):
            cfgs = [gws.merge(c, ssl=True) for c in cfgs]
        self.items = self.create_children(main.Object, cfgs)

    def activate(self):
        self.root.app.register_web_middleware('cors', self.cors_middleware)

    def cors_middleware(self, req: gws.IWebRequester, nxt):
        cors = req.site.corsOptions
        if not cors:
            return nxt()

        if req.method == 'OPTIONS':
            res = req.content_responder(gws.ContentResponse(content='', mime='text/plain'))
        else:
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

        for s in self.items:
            if s.host.lower() == host:
                return s
        for s in self.items:
            if s.host == '*':
                return s

        # there must be a '*' site
        raise ValueError('unknown host', host)

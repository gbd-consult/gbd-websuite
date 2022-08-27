import re

import gws
import gws.lib.net
import gws.types as t

from . import core


class Object(gws.Node, gws.IWebSite):
    assetsRoot: t.Optional[gws.DocumentRoot]
    canonicalHost: str
    corsOptions: core.CorsOptions

    host: str
    rewriteRules: t.List[core.RewriteRule]
    ssl: bool
    staticRoot: gws.DocumentRoot

    def configure(self):

        self.host = self.var('host', default='*')
        self.canonicalHost = self.var('canonicalHost')

        self.staticRoot = core.create_document_root(self.var('root'))
        self.assetsRoot = core.create_document_root(self.var('assets'))

        # config.ssl is populated in the application init
        self.ssl = self.var('ssl')

        self.rewriteRules = self.var('rewrite', default=[])
        for r in self.rewriteRules:
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.errorPage = self.create_child(gws.ext.object.template, self.var('errorPage'), optional=True)

        self.corsOptions = core.CorsOptions(
            allow_origin=self.var('cors.allowOrigin'),
            allow_credentials=self.var('cors.allowCredentials'),
            allow_headers=self.var('cors.allowHeaders'),
        )

    def url_for(self, req, path, **params):
        if gws.lib.net.is_abs_url(path):
            return gws.lib.net.add_params(path, params)

        proto = 'https' if self.ssl else 'http'
        host = self.canonicalHost or (req.env('HTTP_HOST') if self.host == '*' else self.host)
        base = proto + '://' + host

        for rule in self.rewriteRules:
            if rule.reversed:
                m = re.match(rule.pattern, path)
                if m:
                    # we use nginx syntax $1, need python's \1
                    t = rule.target.replace('$', '\\')
                    s = re.sub(rule.pattern, t, path)
                    url = s if gws.lib.net.is_abs_url(s) else base + '/' + s.lstrip('/')
                    return gws.lib.net.add_params(url, params)

        url = base + '/' + path.lstrip('/')
        return gws.lib.net.add_params(url, params)

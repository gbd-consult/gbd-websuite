import re

import gws
import gws.lib.net
import gws.types as t
from . import core


class Config(gws.Config):
    """Site (virtual host) configuration"""

    assets: t.Optional[core.DocumentRootConfig]  #: assets location and options
    cors: t.Optional[core.CorsConfig]  #: cors configuration
    errorPage: t.Optional[gws.ext.template.Config]  #: error page template
    host: str = '*'  #: host name
    rewrite: t.Optional[t.List[core.RewriteRule]]  #: rewrite rules
    canonicalHost: str = ''  #: hostname for reversed rewriting
    root: core.DocumentRootConfig  #: document root location and options


class Object(gws.Object, gws.IWebSite):
    assets_root: t.Optional[gws.DocumentRoot]
    canonical_host: str
    cors_options: core.CorsOptions
    error_page: t.Optional[gws.ITemplate]
    host: str
    rewrite_rules: t.List[core.RewriteRule]
    ssl: bool
    static_root: gws.DocumentRoot

    def configure(self):

        self.host = self.var('host', default='*')
        self.canonical_host = self.var('canonicalHost')

        self.static_root = core.document_root_from_config(self.var('root'))
        self.assets_root = core.document_root_from_config(self.var('assets'))

        # config.ssl is populated in the application init
        self.ssl = self.var('ssl')

        self.rewrite_rules = self.var('rewrite', default=[])
        for r in self.rewrite_rules:
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.error_page = self.create_child_if_config('gws.ext.template', self.var('errorPage'))

        self.cors_options = core.CorsOptions(
            allow_origin=self.var('cors.allowOrigin'),
            allow_credentials=self.var('cors.allowCredentials'),
            allow_headers=self.var('cors.allowHeaders'),
        )

    def url_for(self, req, url):
        if gws.lib.net.is_abs_url(url):
            return url

        proto = 'https' if self.ssl else 'http'
        host = self.canonical_host or (req.env('HTTP_HOST') if self.host == '*' else self.host)
        base = proto + '://' + host

        for rule in self.rewrite_rules:
            if rule.reversed:
                m = re.match(rule.pattern, url)
                if m:
                    # we use nginx syntax $1, need python's \1
                    t = rule.target.replace('$', '\\')
                    s = re.sub(rule.pattern, t, url)
                    return s if gws.lib.net.is_abs_url(s) else base + s

        return base + '/' + url.lstrip('/')

import re

import gws
import gws.common.template
import gws.tools.net
import gws.types as t


class RewriteRule(t.Data):
    """Rewrite rule"""

    match: t.regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: t.Optional[dict]  #: additional options


class CorsConfig(t.Config):
    """CORS options"""

    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: t.Optional[t.List[str]]


class Config(t.Config):
    """Site (virtual host) configuration"""

    assets: t.Optional[t.DocumentRootConfig]  #: assets location and options
    cors: t.Optional[CorsConfig]  #: cors configuration
    errorPage: t.Optional[t.ext.template.Config]  #: error page template
    host: str = '*'  #: host name
    reversedUrl: str = ''  #: base url for reversed addresses
    reversedRewrite: t.Optional[t.List[RewriteRule]]  #: reversed rewrite rules
    rewrite: t.Optional[t.List[RewriteRule]]  #: rewrite rules
    root: t.DocumentRootConfig  #: document root location and options


class Object(gws.Object):
    def __init__(self):
        super().__init__()
        self.host = ''
        self.ssl = False
        self.error_page: gws.common.template.Object = None
        self.static_root: t.DocumentRootConfig = None
        self.assets_root: t.DocumentRootConfig = None
        self.rewrite_rules = []
        self.reversed_rewrite_rules = []
        self.cors: CorsConfig = None

    def configure(self):
        super().configure()

        self.host = self.var('host', default='*')
        self.static_root = self.var('root')
        self.assets_root = self.var('assets')

        # config.ssl is populated in the application init
        self.ssl = self.var('ssl')

        self.rewrite_rules = self.var('rewrite', default=[])
        for r in self.rewrite_rules:
            if not gws.tools.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.reversed_rewrite_rules = self.var('reversedRewrite', default=[])
        for r in self.reversed_rewrite_rules:
            r.match = r.match.strip('/')
            # we use nginx syntax $1, need python's \1
            r.target = r.target.replace('$', '\\')

        p = self.var('errorPage')
        if p:
            self.error_page = self.create_object('gws.ext.template', p)

        p = self.var('cors')
        if p and p.get('enabled'):
            self.cors = p

    def url_for(self, req, url):
        if gws.tools.net.is_abs_url(url):
            return url

        proto = 'https' if self.ssl else 'http'
        host = req.env('HTTP_HOST') if self.host == '*' else self.host
        base = proto + '://' + host

        u = url.lstrip('/')

        for rule in self.reversed_rewrite_rules:
            m = re.match(rule.match, u)
            if m:
                s = re.sub(rule.match, rule.target, u)
                if gws.tools.net.is_abs_url(s):
                    return s
                return base + s

        return base + '/' + u

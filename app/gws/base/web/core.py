import re

import gws
import gws.types as t
import gws.lib.net



class DocumentRootConfig(gws.Config):
    """Base directory for assets"""

    dir: gws.DirPath  #: directory path
    allowMime: t.Optional[t.List[str]]  #: allowed mime types
    denyMime: t.Optional[t.List[str]]  #: disallowed mime types (from the standard list)


class DocumentRoot(gws.Node, gws.IDocumentRoot):
    dir: str
    allow_mime: t.List[str]
    deny_mime: t.List[str]

    def configure(self):
        self.dir = self.var('dir')
        self.allow_mime = self.var('allowMime')
        self.deny_mime = self.var('denyMime')


#


class CorsConfig(gws.Config):
    allowOrigin: str = '*'  #: Access-Control-Allow-Origin header
    allowCredentials: bool = False  #: Access-Control-Allow-Credentials header
    allowHeaders: t.Optional[t.List[str]]  #: Access-Control-Allow-Headers header


class CorsOptions(gws.Data):
    allow_origin: str
    allow_credentials: bool
    allow_headers: t.Optional[t.List[str]]


class RewriteRule(gws.Data):
    match: gws.Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: t.Optional[dict]  #: additional options


#

class SiteConfig(gws.Config):
    """Site (virtual host) configuration"""

    assets: t.Optional[DocumentRootConfig]  #: assets location and options
    cors: t.Optional[CorsConfig]  #: cors configuration
    errorPage: t.Optional[gws.ext.template.Config]  #: error page template
    host: str = '*'  #: host name
    rewrite: t.Optional[t.List[RewriteRule]]  #: rewrite rules
    reversedHost: str = ''  #: hostname for reversed rewriting
    reversedRewrite: t.Optional[t.List[RewriteRule]]  #: reversed rewrite rules
    root: DocumentRootConfig  #: document root location and options


class Site(gws.Node, gws.IWebSite):
    host: str
    reversed_host: str
    cors_options: CorsOptions
    static_root: gws.IDocumentRoot
    assets_root: gws.IDocumentRoot
    ssl: bool
    error_page: t.Optional[gws.ITemplate]

    rewrite_rules: t.List[RewriteRule]
    reversed_rewrite_rules: t.List[RewriteRule]

    def configure(self):

        self.host: str = self.var('host', default='*')
        self.reversed_host: str = self.var('reversedHost')

        self.static_root = t.cast(gws.IDocumentRoot, self.create_child(DocumentRoot, self.var('root')))
        self.assets_root = t.cast(gws.IDocumentRoot, self.create_child(DocumentRoot, self.var('assets')))

        # config.ssl is populated in the application init
        self.ssl: bool = self.var('ssl')

        self.rewrite_rules: t.List[RewriteRule] = self.var('rewrite', default=[])
        for r in self.rewrite_rules:
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.reversed_rewrite_rules: t.List[RewriteRule] = self.var('reversedRewrite', default=[])
        for r in self.reversed_rewrite_rules:
            r.match = str(r.match).strip('/')
            # we use nginx syntax $1, need python's \1
            r.target = r.target.replace('$', '\\')

        p = self.var('errorPage')
        self.error_page: t.Optional[gws.ITemplate] = self.root.create_object('gws.ext.template', p) if p else None

        self.cors_options = CorsOptions(
            allow_origin=self.var('cors.allowOrigin'),
            allow_credentials=self.var('cors.allowCredentials'),
            allow_headers=self.var('cors.allowHeaders'),
        )

    def url_for(self, req, url):
        if gws.lib.net.is_abs_url(url):
            return url

        proto = 'https' if self.ssl else 'http'
        host = self.reversed_host or (req.env('HTTP_HOST') if self.host == '*' else self.host)
        base = proto + '://' + host

        u = url.lstrip('/')

        for rule in self.reversed_rewrite_rules:
            m = re.match(rule.match, u)
            if m:
                s = re.sub(rule.match, rule.target, u)
                if gws.lib.net.is_abs_url(s):
                    return s
                return base + s

        return base + '/' + u


#

class SSLConfig(gws.Config):
    """SSL configuration"""

    crt: gws.FilePath  #: crt bundle location
    key: gws.FilePath  #: key file location


class Config(gws.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[SiteConfig]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration


#

DEFAULT_SITE = Config(
    host='*',
    root=DocumentRootConfig(dir='/data/web')
)

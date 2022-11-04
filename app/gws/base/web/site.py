import re

import gws
import gws.lib.net
import gws.types as t


class CorsConfig(gws.Config):
    allowMethods: t.Optional[t.List[str]] 
    """Access-Control-Allow-Methods header"""
    allowOrigin: str = '*' 
    """Access-Control-Allow-Origin header"""
    allowCredentials: bool = False 
    """Access-Control-Allow-Credentials header"""
    allowHeaders: t.Optional[t.List[str]] 
    """Access-Control-Allow-Headers header"""


class RewriteRuleConfig(gws.Config):
    pattern: gws.Regex 
    """expression to match the url against"""
    target: str 
    """target url with placeholders"""
    options: t.Optional[dict] 
    """additional options"""
    reversed: bool = False 
    """reversed rewrite rule"""


class SSLConfig(gws.Config):
    """SSL configuration"""

    crt: gws.FilePath 
    """crt bundle location"""
    key: gws.FilePath 
    """key file location"""


class DocumentRootConfig(gws.Config):
    """Base directory for assets"""

    dir: gws.DirPath 
    """directory path"""
    allowMime: t.Optional[t.List[str]] 
    """allowed mime types"""
    denyMime: t.Optional[t.List[str]] 
    """disallowed mime types (from the standard list)"""


class Config(gws.Config):
    """Site (virtual host) configuration"""

    assets: t.Optional[DocumentRootConfig] 
    """assets location and options"""
    cors: t.Optional[CorsConfig] 
    """cors configuration"""
    errorPage: t.Optional[gws.ext.config.template] 
    """error page template"""
    host: str = '*' 
    """host name"""
    rewrite: t.Optional[t.List[RewriteRuleConfig]] 
    """rewrite rules"""
    canonicalHost: str = '' 
    """hostname for reversed rewriting"""
    root: DocumentRootConfig 
    """document root location and options"""


class Object(gws.Node, gws.IWebSite):
    canonicalHost: str
    ssl: bool

    def configure(self):

        self.host = self.var('host', default='*')
        self.canonicalHost = self.var('canonicalHost')

        self.staticRoot = gws.WebDocumentRoot(self.var('root'))

        p = self.var('assets')
        self.assetsRoot = gws.WebDocumentRoot(p) if p else None

        self.ssl = self.var('ssl')

        self.rewriteRules = self.var('rewrite', default=[])
        for r in self.rewriteRules:
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.errorPage = self.create_child_if_configured(gws.ext.object.template, self.var('errorPage'))
        self.corsOptions = self.var('cors')

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

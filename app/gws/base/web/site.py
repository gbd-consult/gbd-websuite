from typing import Optional

import re

import gws
import gws.lib.net


class CorsConfig(gws.Config):
    """CORS configuration."""

    allowCredentials: bool = False
    """Access-Control-Allow-Credentials header."""
    allowHeaders: str = ''
    """Access-Control-Allow-Headers header."""
    allowMethods: str = ''
    """Access-Control-Allow-Methods header."""
    allowOrigin: str = ''
    """Access-Control-Allow-Origin header."""
    maxAge: int = 5
    """Access-Control-Max-Age header."""


class RewriteRuleConfig(gws.Config):
    """Rewrite rule configuration."""

    pattern: gws.Regex
    """Expression to match the url against."""
    target: str
    """Target url with placeholders."""
    options: Optional[dict]
    """Additional options."""
    reversed: bool = False
    """Reversed rewrite rule."""


class SSLConfig(gws.Config):
    """SSL configuration."""

    crt: gws.FilePath
    """Crt bundle location."""
    key: gws.FilePath
    """Key file location."""


class WebDocumentRootConfig(gws.Config):
    """Web-accessible directory."""

    dir: gws.DirPath
    """Directory path."""
    allowMime: Optional[list[str]]
    """Allowed mime types."""
    denyMime: Optional[list[str]]
    """Disallowed mime types (from the standard list)."""


class Config(gws.Config):
    """Site (virtual host) configuration"""

    assets: Optional[WebDocumentRootConfig]
    """Root directory for assets."""
    cors: Optional[CorsConfig]
    """Cors configuration."""
    errorPage: Optional[gws.ext.config.template]
    """Error page template."""
    host: str = '*'
    """Host name."""
    rewrite: Optional[list[RewriteRuleConfig]]
    """Rewrite rules."""
    canonicalHost: str = ''
    """Hostname for reversed URL rewriting."""
    root: WebDocumentRootConfig
    """Root directory for static documents."""


class Object(gws.WebSite):
    canonicalHost: str
    ssl: bool

    def configure(self):

        self.host = self.cfg('host', default='*')
        self.canonicalHost = self.cfg('canonicalHost')

        self.staticRoot = gws.WebDocumentRoot(self.cfg('root'))

        p = self.cfg('assets')
        self.assetsRoot = gws.WebDocumentRoot(p) if p else None

        self.ssl = self.cfg('ssl')

        self.rewriteRules = self.cfg('rewrite', default=[])
        for r in self.rewriteRules:
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')

        self.errorPage = self.create_child_if_configured(gws.ext.object.template, self.cfg('errorPage'))
        self.corsOptions = self.cfg('cors')

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

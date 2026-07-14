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
    hsts: gws.Duration = '365d'
    """HSTS max age."""


class WebDirConfig(gws.Config):
    """Web-accessible directory."""

    dir: gws.DirPath
    """Directory path."""
    allowMime: Optional[list[str]]
    """Allowed mime types."""
    denyMime: Optional[list[str]]
    """Disallowed mime types (from the standard list)."""


class Config(gws.Config):
    """Site (virtual host) configuration"""

    assets: Optional[WebDirConfig]
    """Root directory for assets."""
    cors: Optional[CorsConfig]
    """Cors configuration."""
    contentSecurityPolicy: str = "default-src 'self'; img-src * data: blob:"
    """Content Security Policy for this site."""
    permissionsPolicy: str = 'geolocation=(self), camera=(), microphone=()'
    """Permissions Policy for this site."""
    errorPage: Optional[gws.ext.config.template]
    """Error page template. (deprecated in 8.4)"""
    host: str = '*'
    """Host name this site responds to (asterisk for any)."""
    rewrite: Optional[list[RewriteRuleConfig]]
    """Rewrite rules. (deprecated in 8.4)"""
    rewriteRules: Optional[list[RewriteRuleConfig]]
    """Rewrite rules. (added in 8.4)"""
    withDefaultRewriteRules: bool = True
    """Whether to add default rewrite rules. (added in 8.4)"""
    canonicalHost: str = ''
    """Hostname for reversed URL rewriting."""
    useForwardedHost: bool = False
    """Use  X-Forwarded-Host for host matching."""
    root: Optional[WebDirConfig]
    """Root directory for static documents."""


DEFAULT_ASSETS_DIR = '/data/assets'
DEFAULT_WEB_DIR = '/data/web'
DEFAULT_REWRITE_RULES = [
    gws.WebRewriteRule(pattern=r'^/$', target='/_/webPage/name/home'),
    gws.WebRewriteRule(pattern=r'^/project/([a-z0-9_-]+)$', target='/_/webPage/name/project/projectUid/$1'),
]


class Object(gws.WebSite):
    canonicalHost: str
    ssl: bool
    contentSecurityPolicy: str
    permissionsPolicy: str
    useForwardedHost: bool

    def configure(self):
        self.host = self.cfg('host', default='*')
        self.canonicalHost = self.cfg('canonicalHost')
        self.useForwardedHost = self.cfg('useForwardedHost')
        self.ssl = self.cfg('ssl')
        self.corsOptions = self.cfg('cors')
        self.contentSecurityPolicy = self.cfg('contentSecurityPolicy')
        self.permissionsPolicy = self.cfg('permissionsPolicy')
        # deprecated
        self.errorPage = self.create_child_if_configured(gws.ext.object.template, self.cfg('errorPage'))

        p = self.cfg('root')
        if p:
            self.staticRoot = gws.WebDocumentRoot(p)
        elif gws.u.is_dir(DEFAULT_WEB_DIR):
            self.staticRoot = gws.WebDocumentRoot(dir=DEFAULT_WEB_DIR)
        else:
            # note: web root must exist
            gws.log.warning(f'web root {DEFAULT_WEB_DIR!r} does not exist, using temporary directory')
            self.staticRoot = gws.WebDocumentRoot(dir=gws.u.ensure_dir(gws.c.TMP_DIR + '/web'))

        p = self.cfg('assets')
        if p:
            self.assetsRoot = gws.WebDocumentRoot(p)
        elif gws.u.is_dir(DEFAULT_ASSETS_DIR):
            self.assetsRoot = gws.WebDocumentRoot(dir=DEFAULT_ASSETS_DIR)
        else:
            # note: assets root is optional
            self.assetsRoot = None

        self.rewriteRules = []
        p = self.cfg('rewriteRules')
        if not p:
            # deprecated
            p = self.cfg('rewrite')
        if not p:
            p = []
        for c in p:
            r = gws.WebRewriteRule(c)
            if not gws.lib.net.is_abs_url(r.target):
                # ensure rewriting from root
                r.target = '/' + r.target.lstrip('/')
            self.rewriteRules.append(r)

        if self.cfg('withDefaultRewriteRules'):
            patterns = set(r.pattern for r in self.rewriteRules)
            for c in DEFAULT_REWRITE_RULES:
                if c.pattern not in patterns:
                    self.rewriteRules.insert(0, c)

    def url_for(self, req, path, **params):
        if gws.lib.net.is_abs_url(path):
            return gws.lib.net.add_params(path, params)

        proto = 'https' if self.ssl else 'http'
        if self.canonicalHost:
            host = self.canonicalHost
        elif self.host != '*':
            host = self.host
        else:
            host, proto = self._host_proto_from_env(req.environ)

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

    def match_host(self, environ):
        if self.host == '*':
            return True
        host, _ = self._host_proto_from_env(environ)
        lh = host.lower().split(':')[0].strip()
        return lh == self.host.lower()

    def _host_proto_from_env(self, environ):
        host = environ.get('HTTP_HOST', '')
        proto = 'https' if self.ssl else 'http'
        if not self.useForwardedHost:
            return host, proto
        fh = environ.get('HTTP_X_FORWARDED_HOST', '').split(',')[0].strip()
        fp = environ.get('HTTP_X_FORWARDED_PROTO', '').split(',')[0].strip().lower()
        return fh or host, fp or proto

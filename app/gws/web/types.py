import gws.types as t


class SSLConfig(t.Config):
    """SSL configuration"""

    crt: t.filepath  #: crt file location
    key: t.filepath  #: key file location


class RewriteRule(t.Data):
    """Rewrite rule"""

    match: t.regex  #: expression to match the url against
    target: t.formatstr  #: target url with placeholders
    options: t.Optional[dict]  #: additional options


class CorsConfig(t.Config):
    """CORS options"""

    enabled: bool = False
    allowOrigin: str = '*'
    allowCredentials: bool = False
    allowHeaders: t.Optional[t.List[str]]


class SiteConfig(t.Config):
    """Site (virtual host) configuration"""

    host: str = '*'  #: host name
    rewrite: t.Optional[t.List[RewriteRule]]  #: rewrite rules
    cors: t.Optional[CorsConfig]  #: cors configuration
    root: t.DocumentRootConfig  #: document root location and options
    assets: t.Optional[t.DocumentRootConfig]  #: assets location and options


class Config(t.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[SiteConfig]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration

import gws
import gws.types as t


class DocumentRootConfig(gws.Config):
    """Base directory for assets"""

    dir: gws.DirPath  #: directory path
    allowMime: t.Optional[t.List[str]]  #: allowed mime types
    denyMime: t.Optional[t.List[str]]  #: disallowed mime types (from the standard list)


class CorsConfig(gws.Config):
    allowOrigin: str = '*'  #: Access-Control-Allow-Origin header
    allowCredentials: bool = False  #: Access-Control-Allow-Credentials header
    allowHeaders: t.Optional[t.List[str]]  #: Access-Control-Allow-Headers header


class CorsOptions(gws.Data):
    allow_origin: str
    allow_credentials: bool
    allow_headers: t.Optional[t.List[str]]


class RewriteRule(gws.Data):
    pattern: gws.Regex  #: expression to match the url against
    target: str  #: target url with placeholders
    options: t.Optional[dict]  #: additional options
    reversed: bool = False  #: reversed rewrite rule


class SSLConfig(gws.Config):
    """SSL configuration"""

    crt: gws.FilePath  #: crt bundle location
    key: gws.FilePath  #: key file location


class SiteConfig(gws.Config):
    """Site (virtual host) configuration"""

    assets: t.Optional[DocumentRootConfig]  #: assets location and options
    cors: t.Optional[CorsConfig]  #: cors configuration
    errorPage: t.Optional[gws.ext.config.template]  #: error page template
    host: str = '*'  #: host name
    rewrite: t.Optional[t.List[RewriteRule]]  #: rewrite rules
    canonicalHost: str = ''  #: hostname for reversed rewriting
    root: DocumentRootConfig  #: document root location and options


class Config(gws.Config):
    """Web server configuration"""

    sites: t.Optional[t.List[SiteConfig]]  #: configured sites
    ssl: t.Optional[SSLConfig]  #: ssl configuration



def create_document_root(cfg: t.Optional[DocumentRootConfig]) -> t.Optional[gws.DocumentRoot]:
    if not cfg:
        return None

    cfg = DocumentRootConfig(cfg)
    return gws.DocumentRoot(
        dir=cfg.dir,
        allow_mime=cfg.allowMime,
        deny_mime=cfg.denyMime)

import gws
import gws.types as t


class DocumentRootConfig(gws.Config):
    """Base directory for assets"""

    dir: gws.DirPath  #: directory path
    allowMime: t.Optional[t.List[str]]  #: allowed mime types
    denyMime: t.Optional[t.List[str]]  #: disallowed mime types (from the standard list)


def document_root_from_config(cfg: t.Optional[DocumentRootConfig]) -> t.Optional[gws.DocumentRoot]:
    if not cfg:
        return None

    cfg = DocumentRootConfig(cfg)
    return gws.DocumentRoot(
        dir=cfg.dir,
        allow_mime=cfg.allowMime,
        deny_mime=cfg.denyMime)


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


class SSLConfig(gws.Config):
    """SSL configuration"""

    crt: gws.FilePath  #: crt bundle location
    key: gws.FilePath  #: key file location

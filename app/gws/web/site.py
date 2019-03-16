import gws
import gws.common.template
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
    errorPage: t.Optional[t.TemplateConfig]  #: error page template
    host: str = '*'  #: host name
    reversedBase: str = ''  #: reversed base address
    reversedRewrite: t.Optional[t.List[RewriteRule]]  #: reversed rewrite rules
    rewrite: t.Optional[t.List[RewriteRule]]  #: rewrite rules
    root: t.DocumentRootConfig  #: document root location and options


class Object(gws.PublicObject):
    def __init__(self):
        super().__init__()
        self.host = ''
        self.error_page: gws.common.template.Object = None
        self.static_root: t.DocumentRootConfig = None
        self.assets_root: t.DocumentRootConfig = None
        self.rewrite_rules = []
        self.cors = None

    def configure(self):
        super().configure()
        self.host = self.var('host', default='*')

        self.static_root = self.var('root')
        self.assets_root = self.var('assets')

        self.rewrite_rules = self.var('rewrite', default=[])
        for r in self.rewrite_rules:
            if not r.target.startswith('/'):
                r.target = '/' + r.target

        p = self.var('errorPage')
        if p:
            self.error_page = self.create_object('gws.ext.template', p)

        p = self.var('cors')
        if p and p.get('enabled'):
            self.cors = p

    # reversedBase: str = ''  #: reversed base address
    # reversedRewrite: t.Optional[t.List[RewriteRule]]  #: reversed rewrite rules
    # cors: t.Optional[CorsConfig]  #: cors configuration
    # root: DocumentRootConfig  #: document root location and options
    # assets: t.Optional[DocumentRootConfig]  #: assets location and options
    # errorPage: t.Optional[t.TemplateConfig]  #: error page template

    # if not s.get('reversedBase'):
    #     s.reversedBase = environ['wsgi.url_scheme'] + '://' + environ['HTTP_HOST']
    # if not s.get('reversedRewrite'):
    #     s.reversedRewrite = []

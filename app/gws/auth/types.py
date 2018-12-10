import gws.types as t


class CookieConfig(t.Config):
    """session cookie parameters"""

    name: str = 'auth'  #: name for the cookie
    path: str = '/'  #: cookie path


class SessionConfig(t.Config):
    """session configuration"""

    lifeTime: t.duration = 1200  #: session life time
    heartBeat: bool = False  #: refresh sessions automatically
    storage: str = 'sqlite'  #: session storage engine


class Config(t.Config):
    """authentication and authorization options"""

    httpEnabled: bool = True  #: http authorization enabled
    httpsOnly: bool = False  #: http authorization via ssl only
    cookie: t.Optional[CookieConfig] = {}  #: session cookie parameters
    session: t.Optional[SessionConfig] = {}  #: session configuration
    providers: t.List[t.ext.auth.provider.Config]  #: authorization providers

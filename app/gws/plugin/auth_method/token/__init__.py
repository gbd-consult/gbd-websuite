"""HTTP Token authorisation method.

The token authorization works by passing a token in an HTTP header.
For example, with this configuration::

    auth.methods+ {
        type "token"
        header "X-My-Auth"
        prefix "Bearer"

    }

the application would expect a header like ``X-My-Auth: Bearer <token>``, extract the token value
and pass it along to authorization providers.
"""

import gws
import gws.base.auth
import gws.base.web
import gws.types as t

gws.ext.new.authMethod('token')


class Config(gws.base.auth.method.Config):
    """HTTP-token authorization options"""

    header: str
    """HTTP header name"""
    prefix: str = ''
    """token prefix"""


class Object(gws.base.auth.method.Object):
    header: str
    prefix: str

    def configure(self):
        self.uid = 'gws.plugin.auth_method.token'
        self.header = self.cfg('header')
        self.prefix = self.cfg('prefix', default='')
        self.root.app.register_middleware(self.uid, self, depends_on=['auth'])

    ##

    def enter_middleware(self, req):
        pass

    def exit_middleware(self, req, res):
        pass

    def open_session(self, req):
        credentials = self._parse_header(req)
        if not credentials:
            return
        try:
            user = self.authMgr.authenticate(self, credentials)
        except gws.ForbiddenError as exc:
            raise gws.base.web.error.Forbidden() from exc
        if user:
            user.authToken = credentials.get('token')
            return self.authMgr.sessionMgr.create(self, user)

    def close_session(self, req, res):
        pass

    def _parse_header(self, req: gws.IWebRequester):
        h = req.header(self.header)
        if not h:
            return

        a = h.strip().split()

        if self.prefix:
            if len(a) != 2 or a[0].lower() != self.prefix.lower():
                return
            return gws.Data(token=a[1])
        else:
            if len(a) != 1:
                return
            return gws.Data(token=a[0])

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


gws.ext.new.authMethod('token')


class Config(gws.base.auth.method.Config):
    """HTTP-token authorization options (added in 8.1)"""

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

    ##

    def open_session(self, req):
        am = self.root.app.authMgr
        credentials = self._parse_header(req)
        if not credentials:
            return
        user = am.authenticate(self, credentials)
        if user:
            user.authToken = credentials.get('token')
            return am.sessionMgr.create(self, user)

    def close_session(self, req, res):
        pass

    def _parse_header(self, req: gws.WebRequester):
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

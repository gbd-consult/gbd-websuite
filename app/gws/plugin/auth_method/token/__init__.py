"""HTTP token authentication method.

The client passes a token in an HTTP header. The method reads the configured
header, checks the optional prefix and passes the token to the authentication
providers as ``token`` credentials. For an authenticated user, the token is
stored in ``user.authToken`` and a transient session is opened. No cookie is
set.

The prefix is compared case-insensitively. Without a prefix, the header value
must be the token alone.

Example::

    auth.methods+ {
        type "token"
        header "X-My-Auth"
        prefix "Bearer"
    }

With this configuration, the application expects a header like
``X-My-Auth: Bearer <token>``.
"""

import gws
import gws.base.auth
import gws.base.web


@gws.ext.config.authMethod('token')
class Config(gws.base.auth.method.Config):
    """Authentication with a token passed in an HTTP header."""

    header: str
    """HTTP header that carries the token."""
    prefix: str = ''
    """Prefix expected before the token in the header value."""


@gws.ext.object.authMethod('token')
class Object(gws.base.auth.method.Object):
    """HTTP token authentication method."""

    header: str
    """Name of the HTTP header that carries the token."""
    prefix: str
    """Prefix expected before the token, empty if none."""

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
        user = am.authenticate(self, credentials, req)
        if user:
            user.authToken = credentials.get('token')
            return am.create_transient_session(self, user)

    def _parse_header(self, req: gws.WebRequester):
        """Extract the token from the configured header."""
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

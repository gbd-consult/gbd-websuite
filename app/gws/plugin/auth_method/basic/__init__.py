"""HTTP Basic authentication method.

The client sends the credentials with every request in an
``Authorization: Basic ...`` header. The method decodes the header into
``username`` and ``password`` credentials, passes them to the authentication
providers and opens a transient session for the authenticated user. No cookie
is set.

The method is registered as a middleware after ``auth``. If a GET request is
denied with status ``403``, the response is changed to ``401`` with a
``WWW-Authenticate`` header, so that browsers show a login dialog.

Example::

    auth.methods+ {
        type "basic"
        realm "My Application"
    }
"""

from typing import Optional

import gws
import gws.base.auth
import gws.base.web
import gws.lib.text


@gws.ext.config.authMethod('basic')
class Config(gws.base.auth.method.Config):
    """HTTP basic authentication."""

    realm: Optional[str]
    """Authentication realm sent to the client."""


@gws.ext.object.authMethod('basic')
class Object(gws.base.auth.method.Object):
    """HTTP Basic authentication method."""

    realm: str
    """Authentication realm sent in the ``WWW-Authenticate`` header."""

    def configure(self):
        self.uid = 'gws.plugin.auth_method.basic'
        self.realm = self.cfg('realm', default='Restricted Area')
        self.root.app.middlewareMgr.register(self, self.uid, depends_on=['auth'])

    ##

    def exit_middleware(self, req, res):
        if res.status == 403 and req.isGet:
            res.set_status(401)
            res.add_header('WWW-Authenticate', f'Basic realm={self.realm}, charset="UTF-8"')
            gws.log.debug(f'auth basic: redirect {res.status=}')


    def open_session(self, req):
        am = self.root.app.authMgr
        credentials = self._parse_header(req)
        if not credentials:
            return
        user = am.authenticate(self, credentials, req)
        if user:
            return am.create_transient_session(self, user)

    def _parse_header(self, req: gws.WebRequester):
        """Extract the username and password from the ``Authorization`` header."""
        h = req.header('Authorization')
        if not h:
            return

        a = h.strip().split()
        if len(a) != 2 or a[0].lower() != 'basic':
            return

        try:
            b = gws.u.to_str(gws.lib.text.from_base64(a[1]))
        except gws.lib.text.Error:
            return

        c = b.split(':', 1)
        if len(c) != 2:
            return

        username = c[0].strip()
        if not username:
            return

        return gws.Data(username=username, password=c[1])

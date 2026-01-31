"""HTTP Basic authorisation method."""

from typing import Optional

import base64

import gws
import gws.base.auth
import gws.base.web

gws.ext.new.authMethod('basic')


class Config(gws.base.auth.method.Config):
    """HTTP-basic authorization options"""

    realm: Optional[str]
    """Authentication realm."""


class Object(gws.base.auth.method.Object):
    realm: str

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
        user = am.authenticate(self, credentials)
        if user:
            return am.sessionMgr.create(self, user)

    def close_session(self, req, res):
        pass

    def _parse_header(self, req: gws.WebRequester):
        h = req.header('Authorization')
        if not h:
            return

        a = h.strip().split()
        if len(a) != 2 or a[0].lower() != 'basic':
            return

        try:
            b = gws.u.to_str(base64.decodebytes(gws.u.to_bytes(a[1])))
        except ValueError:
            return

        c = b.split(':')
        if len(c) != 2:
            return

        username = c[0].strip()
        if not username:
            return

        return gws.Data(username=username, password=c[1])

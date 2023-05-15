"""HTTP Basic authorisation method."""

import base64

import gws
import gws.base.auth
import gws.base.web
import gws.types as t

gws.ext.new.authMethod('basic')


class Config(gws.base.auth.method.Config):
    """HTTP-basic authorization options"""

    realm: t.Optional[str]
    """authentication realm"""


class Object(gws.base.auth.method.Object):
    realm: str

    def configure(self):
        self.uid = 'gws.plugin.auth_method.basic'
        self.realm = self.cfg('realm', default='Restricted Area')
        self.root.app.register_middleware(self.uid, self, depends_on=['auth'])

    ##

    def enter_middleware(self, req):
        pass

    def exit_middleware(self, req, res):
        if res.status == 403 and req.isGet:
            res.set_status(401)
            res.add_header('WWW-Authenticate', f'Basic realm={self.realm}, charset="UTF-8')

    def open_session(self, req):
        credentials = _parse_header(req)
        if not credentials:
            return
        try:
            user = self.authMgr.authenticate(self, credentials)
        except gws.ForbiddenError as exc:
            raise gws.base.web.error.Forbidden() from exc
        if user:
            return self.authMgr.sessionMgr.create(self, user)

    def close_session(self, req, res):
        pass


##


def _parse_header(req: gws.IWebRequester):
    h = req.header('Authorization')
    if not h:
        return

    a = h.strip().split()
    if len(a) != 2 or a[0].lower() != 'basic':
        return

    try:
        b = gws.to_str(base64.decodebytes(gws.to_bytes(a[1])))
    except ValueError:
        return

    c = b.split(':')
    if len(c) != 2:
        return

    username = c[0].strip()
    if not username:
        return

    return gws.Data(username=username, password=c[1])

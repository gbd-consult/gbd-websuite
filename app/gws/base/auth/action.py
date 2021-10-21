"""Check user logins and logouts."""

import gws
import gws.types as t

import gws.base.api
import gws.base.web.error

from . import wsgi
from . import user as user_api


class Response(gws.Response):
    user: user_api.Props


class LoginParams(gws.Params):
    username: str
    password: str


@gws.ext.Object('action.auth')
class Object(gws.base.api.action.Object):

    @gws.ext.command('api.auth.check')
    def check(self, req: gws.IWebRequest, p: gws.Params) -> Response:
        """Check the authorization status"""

        return self._resp(req)

    @gws.ext.command('api.auth.login')
    def login(self, req: gws.IWebRequest, p: LoginParams) -> Response:
        """Perform a login"""

        if not req.user.is_guest:
            gws.log.error('login while logged-in')
            raise gws.base.web.error.Forbidden()

        wr = t.cast(wsgi.WebRequest, req)
        web_method = wr.auth.get_method(ext_type='web')
        if not web_method:
            gws.log.error('web method not configured')
            raise gws.base.web.error.Forbidden()

        new_session = web_method.login(wr.auth, p, req)
        if not new_session:
            raise gws.base.web.error.Forbidden()

        wr.session = new_session
        return self._resp(req)

    @gws.ext.command('api.auth.logout')
    def logout(self, req: gws.IWebRequest, p: gws.Params) -> Response:
        """Perform a logout"""

        if req.user.is_guest:
            return self._resp(req)

        wr = t.cast(wsgi.WebRequest, req)
        web_method = wr.auth.get_method(ext_type='web')
        if not web_method:
            gws.log.error('web method not configured')
            raise gws.base.web.error.Forbidden()

        session = wr.session
        if session.method != web_method:
            gws.log.error(f'wrong method for logout: {session.method!r}')
            raise gws.base.web.error.Forbidden()

        wr.session = web_method.logout(wr.auth, session, req)
        return self._resp(req)

    def _resp(self, req):
        return Response(user=gws.props(req.user, req.user))

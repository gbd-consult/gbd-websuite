"""Check user logins and logouts."""

import gws
import gws.base.action
import gws.base.web.error
import gws.types as t

from . import wsgi, user


class Response(gws.Response):
    user: user.Props


class LoginParams(gws.Request):
    username: str
    password: str


@gws.ext.object.action('auth')
class Object(gws.base.action.Object):

    @gws.ext.command.api('authCheck')
    def check(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        """Check the authorization status"""

        return self._resp(req)

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.IWebRequester, p: LoginParams) -> Response:
        """Perform a login"""

        if not req.user.isGuest:
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

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        """Perform a logout"""

        if req.user.isGuest:
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

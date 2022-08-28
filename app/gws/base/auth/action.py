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

        return self._response(req)

    @gws.ext.command.api('authLogin')
    def login(self, req: gws.IWebRequester, p: LoginParams) -> Response:
        """Perform a login"""

        self.root.app.auth.login(p, req)
        return self._response(req)

    @gws.ext.command.api('authLogout')
    def logout(self, req: gws.IWebRequester, p: gws.Request) -> Response:
        """Perform a logout"""

        self.root.app.auth.logout(req)
        return self._response(req)


    def _response(self, req):
        return Response(user=gws.props(req.user, req.user))

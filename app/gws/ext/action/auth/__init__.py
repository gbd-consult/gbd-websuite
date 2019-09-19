import gws
import gws.auth.api
import gws.web
import gws.types as t
import gws.auth.user


class Config(t.WithTypeAndAccess):
    """Authorization action"""
    pass


class Response(t.Response):
    user: gws.auth.user.Props


class LoginParams(t.Params):
    username: str
    password: str


class LogoutParams(t.Params):
    pass


class CheckParams(t.Params):
    pass


def _feedback(req: gws.web.AuthRequest):
    return Response({'user': req.user.props_for(req.user)})


class Object(gws.ActionObject):

    def api_check(self, req, p: CheckParams) -> Response:
        """Check the authorization status"""

        return _feedback(req)

    def api_login(self, req, p: LoginParams) -> Response:
        """Perform a login"""

        if not req.user.is_guest:
            gws.log.error('login while logged-in')
            raise gws.web.error.Forbidden()

        try:
            user = gws.auth.api.authenticate_user(p.username, p.password)
        except gws.auth.api.Error as err:
            raise gws.web.error.Forbidden() from err

        if not user:
            raise gws.web.error.Forbidden()

        req.logged_in(user)
        return _feedback(req)

    def api_logout(self, req, p: LogoutParams) -> Response:
        """Perform a logout"""

        req.logged_out()
        return _feedback(req)

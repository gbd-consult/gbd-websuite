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


class Object(gws.ActionObject):

    def api_check(self, req: gws.web.AuthRequest, p: t.NoParams) -> Response:
        """Check the authorization status"""

        return _feedback(req)

    def api_login(self, req: gws.web.AuthRequest, p: LoginParams) -> Response:
        """Perform a login"""

        req.login(p.username, p.password)
        return _feedback(req)

    def api_logout(self, req: gws.web.AuthRequest, p: t.NoParams) -> Response:
        """Perform a logout"""

        req.logout()
        return _feedback(req)


def _feedback(req: gws.web.AuthRequest):
    return Response({'user': req.user.props})

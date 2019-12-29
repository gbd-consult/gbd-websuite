import gws.auth.user
import gws.web
import gws.types as t

class Config(t.WithTypeAndAccess):
    """Authorization action"""
    pass


class Response(t.Response):
    user: gws.auth.user.Props


class LoginParams(t.Params):
    username: str
    password: str


class Object(gws.ActionObject):

    def api_check(self, req: t.WebRequest, p: t.NoParams) -> Response:
        """Check the authorization status"""

        return _feedback(req)

    def api_login(self, req: t.WebRequest, p: LoginParams) -> Response:
        """Perform a login"""

        req.login(p.username, p.password)
        return _feedback(req)

    def api_logout(self, req: t.WebRequest, p: t.NoParams) -> Response:
        """Perform a logout"""

        req.logout()
        return _feedback(req)


def _feedback(req: t.WebRequest):
    return Response({'user': req.user.props})

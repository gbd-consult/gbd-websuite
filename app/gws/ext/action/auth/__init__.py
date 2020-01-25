import gws.web
import gws.types as t

class Config(t.WithTypeAndAccess):
    """Authorization action"""
    pass


class Response(t.Response):
    user: t.UserProps


class LoginParams(t.Params):
    username: str
    password: str


class Object(gws.ActionObject):

    def api_check(self, req: t.IRequest, p: t.Params) -> Response:
        """Check the authorization status"""

        return _feedback(req)

    def api_login(self, req: t.IRequest, p: LoginParams) -> Response:
        """Perform a login"""

        req.login(p.username, p.password)
        return _feedback(req)

    def api_logout(self, req: t.IRequest, p: t.Params) -> Response:
        """Perform a logout"""

        req.logout()
        return _feedback(req)


def _feedback(req: t.IRequest):
    return Response({'user': req.user.props})

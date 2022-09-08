"""Check user logins and logouts."""

import gws.common.action
import gws.common.template
import gws.web.error
import gws.ext.auth.method.web

from gws.ext.auth.method.web import ActionResult

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Authorization action"""

    templates: t.Optional[t.List[t.ext.template.Config]]  #: client templates


class Response(t.Response):
    actionResult: t.Optional[str]
    user: t.UserProps
    mfaOptions: t.Optional[dict]
    view: t.Optional[str]


class LoginParams(t.Params):
    username: str
    password: str


class MfaVerifyParams(t.Params):
    token: str


class Object(gws.common.action.Object):

    def configure(self):
        super().configure()
        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'))

    def api_check(self, req: t.IRequest, p: t.Params) -> Response:
        """Check the authorization status"""

        return self.response(self.web().action_check(req))

    def api_login(self, req: t.IRequest, p: LoginParams) -> Response:
        """Perform a login"""

        return self.response(self.web().action_login(req, p))

    def api_logout(self, req: t.IRequest, p: t.Params) -> Response:
        """Perform a logout"""

        return self.response(self.web().action_logout(req))

    def api_mfa_verify(self, req: t.IRequest, p: MfaVerifyParams) -> Response:
        """Verify an MFA token"""

        return self.response(self.web().action_mfa_verify(req, p))

    def api_mfa_restart(self, req: t.IRequest, p: t.Params) -> Response:
        """Regenerate an MFA token"""

        return self.response(self.web().action_mfa_restart(req))

    def web(self) -> gws.ext.auth.method.web.Object:
        return t.cast(
            gws.ext.auth.method.web.Object,
            self.root.application.auth.get_method('web'))

    def response(self, r):
        res = Response(actionResult=r.result, user=r.user.props, status=200)

        if r.result in {ActionResult.loginFailed, ActionResult.loginFatal, ActionResult.mfaFatal}:
            res.status = 403

        if r.result in {ActionResult.mfaFailed}:
            res.status = 409

        args = {
            'user': r.user,
            'actionResult': r.result,
        }

        if r.mf:
            args['mfa'] = t.Data(
                uid=r.mf.uid,
                restartCount=r.mf.restartCount,
                verifyCount=r.mf.verifyCount,
            )

        tpl = gws.common.template.find(self.templates, subject='client')
        if tpl:
            res.view = tpl.render(args).content

        return res

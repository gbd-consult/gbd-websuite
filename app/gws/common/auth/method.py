import gws
import gws.types as t

from . import error


#:export IAuthMethod
class Object(gws.Object, t.IAuthMethod):
    type: str

    def configure(self):
        super().configure()
        self.type = self.var('type')

    def open_session(self, auth: t.IAuthManager, req: t.IRequest) -> t.Optional[t.ISession]:
        pass

    def close_session(self, auth: t.IAuthManager, sess: t.ISession, req: t.IRequest, res: t.IResponse):
        pass

    def login(self, auth: t.IAuthManager, login: str, password: str, req: t.IRequest) -> t.Optional[t.ISession]:
        raise error.AccessDenied()

    def logout(self, auth: t.IAuthManager, sess: t.ISession, req: t.IRequest) -> t.ISession:
        pass

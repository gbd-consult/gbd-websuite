import gws
import gws.common.auth

import gws.types as t

from . import wrappers, error


#:export IRequest
class Request(wrappers.BaseRequest, t.IRequest):
    session: t.ISession = None

    @property
    def auth(self) -> t.IAuthManager:
        return self.root.application.auth

    @property
    def user(self) -> t.IUser:
        if not self.session:
            raise gws.Error('session not opened')
        return self.session.user

    def auth_open(self):
        try:
            self.session = self.auth.open_session(self)
        except gws.common.auth.Error as e:
            raise error.Forbidden() from e

    def auth_close(self, res: t.IResponse):
        self.session = self.auth.close_session(self.session, self, res)

    def require(self, klass: str, uid: str) -> t.IObject:
        node = self.root.find(klass, uid)
        if not node:
            gws.log.error('require: not found', klass, uid)
            raise error.NotFound()
        if not self.user.can_use(node):
            gws.log.error('require: denied', klass, uid)
            raise error.Forbidden()
        return node

    def require_project(self, uid: str) -> t.IProject:
        return t.cast(t.IProject, self.require('gws.common.project', uid))

    def require_layer(self, uid: str) -> t.ILayer:
        return t.cast(t.ILayer, self.require('gws.ext.layer', uid))

    def acquire(self, klass: str, uid: str) -> t.Optional[t.IObject]:
        obj = self.root.find(klass, uid)
        if obj and self.user.can_use(obj):
            return obj

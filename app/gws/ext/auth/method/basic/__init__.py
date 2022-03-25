import base64

import gws
import gws.common.auth.method
import gws.common.auth.error

import gws.types as t


# @TODO support WWW-Authenticate at some point

class Config(t.WithType):
    """HTTP-basic authorization options"""

    secure: bool = True  #: use only with SSL


class Object(gws.common.auth.method.Object):

    def configure(self):
        super().configure()
        self.secure = self.var('secure')

    def open_session(self, auth: t.IAuthManager, req: t.IRequest):
        if self.secure and not req.is_secure:
            return

        credentials = self._parse_header(req)
        if not credentials:
            return

        user = auth.authenticate(self, credentials[0], credentials[1])
        if user:
            return auth.new_session(type='http-basic', method=self, user=user)

        # if the header is provided, it has to be correct
        raise gws.common.auth.error.LoginNotFound()

    def _parse_header(self, req: t.IRequest):
        h = req.header('Authorization')
        if not h:
            return

        h = h.strip().split()
        if len(h) != 2 or h[0].lower() != 'basic':
            return

        try:
            h = gws.as_str(base64.decodebytes(gws.as_bytes(h[1])))
        except ValueError:
            return

        h = h.split(':', 1)
        if len(h) != 2:
            return

        return h

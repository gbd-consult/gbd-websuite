import base64

import gws
import gws.types as t

from .. import error, manager, method

# @TODO support WWW-Authenticate at some point

@gws.ext.config.authMethod('basic')
class Config(method.Config):
    """HTTP-basic authorization options"""
    pass


@gws.ext.object.authMethod('basic')
class Object(method.Object):

    def open_session(self, auth, req):
        if self.secure and not req.is_secure:
            return

        login_pass = _parse_header(req)
        if not login_pass:
            return

        user = auth.authenticate(self, gws.Data(username=login_pass[0], password=login_pass[1]))
        if user:
            return t.cast(manager.Object, auth).new_session('http-basic', method=self, user=user)

        # if the header is provided, it has to be correct
        raise error.LoginNotFound()


def _parse_header(req: gws.IWebRequest):
    h = req.header('Authorization')
    if not h:
        return

    a = h.strip().split()
    if len(a) != 2 or a[0].lower() != 'basic':
        return

    try:
        b = gws.to_str(base64.decodebytes(gws.to_bytes(a[1])))
    except ValueError:
        return

    c = b.split(':')
    if len(c) != 2:
        return

    return c

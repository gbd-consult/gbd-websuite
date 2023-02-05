import base64

import gws
import gws.types as t

from .. import error, method

gws.ext.new.authMethod('basic')


# @TODO support WWW-Authenticate at some point

class Config(method.Config):
    """HTTP-basic authorization options"""
    pass


class Object(method.Object):

    def open_session(self, req):
        if self.secure and not req.isSecure:
            return False

        login_pass = _parse_header(req)
        if not login_pass:
            return False

        user = self.auth.authenticate(self, gws.Data(username=login_pass[0], password=login_pass[1]))
        if user:
            sess = self.auth.session_create('http-basic', method=self, user=user)
            self.auth.session_activate(req, sess)
            return True

        # if the header is provided, it has to be correct
        raise error.LoginNotFound()

    def close_session(self, req, res):
        self.auth.session_activate(req, None)
        return True


def _parse_header(req: gws.IWebRequester):
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

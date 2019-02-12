import gws.tools.password
import gws.tools.json2
import gws.web
import gws.types.spec
import gws.config.parser

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Remote administration action"""
    pass


class ValidateParams(t.Data):
    password: str
    config: dict


class GetSpecParams(t.Data):
    password: str
    lang: str


_langs = 'en', 'de'


class CheckerError(gws.Error):
    pass


class Checker(gws.types.spec.Validator):
    def error(self, code, msg, val):
        raise CheckerError({
            'errorCode': code,
            'errorMessage': msg,
            'errorPath': '.'.join(str(x) for x in self.keys)
        })


class Object(gws.Object):
    def api_validate(self, req, p: ValidateParams) -> t.Response:
        """Validate configuration"""

        if not _check_password(p.password):
            raise gws.web.error.Forbidden()
        gws.log.info('REMOTE_ADMIN: password accepted')

        spec = gws.config.parser.load_spec('config')
        val = Checker(spec['types'], strict=True)

        try:
            val.get(p.config, 'gws.common.application.Config', 'PP')
        except CheckerError as e:
            r = gws.extend({'ok': False}, e.args[0])
            return t.Response(r)

        return t.Response({'ok': True})

    def api_get_spec(self, req, p: GetSpecParams) -> t.Response:
        """Validate configuration"""

        if not _check_password(p.password):
            raise gws.web.error.Forbidden()
        gws.log.info('REMOTE_ADMIN: password accepted')

        if p.lang not in _langs:
            raise gws.web.error.NotFound()

        spec = {
            'config': gws.config.parser.load_spec('config', p.lang),
            'api': gws.config.parser.load_spec('api', p.lang),
            'cli': gws.config.parser.load_spec('cli', p.lang),
        }

        return t.Response({'spec': spec})


def _read_password():
    try:
        with open(gws.REMOTE_ADMIN_PASSWD_FILE) as fp:
            p = fp.read().strip()
            return p if len(p) > 0 else None
    except:
        return None


def _check_password(password):
    p = _read_password()
    if not p:
        gws.log.warn('REMOTE_ADMIN: error reading the passwd file')
        return False

    ok = gws.tools.password.check(password, p)
    if not ok:
        gws.log.warn('REMOTE_ADMIN: password check error')
        return False

    return True

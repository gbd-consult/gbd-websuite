import gws.tools.password
import gws.tools.json2
import gws.web
import gws.types.spec
import gws.config.spec

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Remote administration action"""
    passwordFile: str = '/gws-var/.remoteadmin'  #: path to the password file


class ValidateParams(t.Params):
    password: str
    config: dict


class GetSpecParams(t.Params):
    password: str
    lang: str


_AVAILABLE_LANGUAGES = 'en', 'de'


class CheckerError(gws.Error):
    pass


class Checker(gws.types.spec.Validator):
    def error(self, code, msg, val):
        raise CheckerError({
            'errorCode': code,
            'errorMessage': msg,
            'errorPath': '.'.join(str(x) for x in self.keys)
        })


class Object(gws.ActionObject):
    @property
    def props(self):
        # no client props for this action
        return None

    def api_validate(self, req, p: ValidateParams) -> t.Response:
        """Validate configuration"""

        if not self._check_password(p.password):
            raise gws.web.error.Forbidden()
        gws.log.info('REMOTE_ADMIN: password accepted')

        spec = gws.config.spec.load('config')
        val = Checker(spec['types'], strict=True)

        try:
            val.get(p.config, 'gws.common.application.Config')
        except CheckerError as e:
            r = gws.extend({'ok': False}, e.args[0])
            return t.Response(r)

        return t.Response({'ok': True})

    def api_get_spec(self, req, p: GetSpecParams) -> t.Response:
        """Get the config spec"""

        if not self._check_password(p.password):
            raise gws.web.error.Forbidden()
        gws.log.info('REMOTE_ADMIN: password accepted')

        if p.lang not in _AVAILABLE_LANGUAGES:
            raise gws.web.error.NotFound()

        spec = {
            'version': gws.VERSION,
            'config': gws.config.spec.load('config', p.lang),
            'api': gws.config.spec.load('api', p.lang),
            'cli': gws.config.spec.load('cli', p.lang),
        }

        return t.Response({'spec': spec})

    def _check_password(self, password):
        p = self._read_password()
        if not p:
            gws.log.warn('REMOTE_ADMIN: error reading the passwd file')
            return False

        ok = gws.tools.password.check(password, p)
        if not ok:
            gws.log.warn('REMOTE_ADMIN: password check error')
            return False

        return True

    def _read_password(self):
        try:
            with open(self.var('passwordFile')) as fp:
                p = fp.read().strip()
                return p if len(p) > 0 else None
        except:
            return None

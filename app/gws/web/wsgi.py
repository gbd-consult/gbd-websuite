"""web application root"""

import gws
import gws.config.loader
import gws.web
import gws.web.action

import gws.types as t

_no_site = t.Data({
    'host': '*'
})


def _site(environ, sites):
    h = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    if not sites:
        return _no_site

    for s in sites:
        if s.host.lower() == h:
            return s
    for s in sites:
        if s.host == '*':
            return s

    gws.log.error('unknown host', h)
    raise gws.web.error.NotFound()


def _with_cors_headers(cors, res):
    if cors.get('allowOrigin'):
        res.headers.add('Access-Control-Allow-Origin', cors.get('allowOrigin'))
    if cors.get('allowCredentials'):
        res.headers.add('Access-Control-Allow-Credentials', 'true')
    if cors.get('allowHeaders'):
        res.headers.add('Access-Control-Allow-Headers', ', '.join(cors.get('allowHeaders')))
    res.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _handle_request(config_root, environ):
    environ['gws.site'] = _site(environ, config_root.application.web_sites)

    req = gws.web.AuthRequest(environ)
    if req.params is None:
        raise gws.web.error.NotFound()

    cors = req.site.get('cors')
    if cors and not cors.get('enabled'):
        cors = None

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response)

    req.auth_begin()

    # gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = gws.web.action.handle(config_root, req)

    req.auth_commit(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def application(environ, start_response):
    config_root = gws.config.root()
    try:
        res = _handle_request(config_root, environ)
        return res(environ, start_response)
    except gws.web.error.HTTPException as e:
        return e(environ, start_response)
    except:
        gws.log.exception()
        return gws.web.error.InternalServerError()(environ, start_response)


gws.config.loader.load()

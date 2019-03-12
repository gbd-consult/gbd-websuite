"""web application root"""

import gws
import gws.config.loader

import gws.web.action
import gws.web.error
import gws.web.auth
import gws.web.wrappers


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


##


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
    req = gws.web.auth.AuthRequest(gws.web.wrappers.add_site(environ, config_root.application.web_sites))
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

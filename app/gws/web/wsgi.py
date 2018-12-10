"""web application root"""

import gws
import gws.config.loader
import gws.web
import gws.web.env
import gws.web.action


def _with_cors_headers(cors, res):
    if cors.get('allowOrigin'):
        res.headers.add('Access-Control-Allow-Origin', cors.get('allowOrigin'))
    if cors.get('allowCredentials'):
        res.headers.add('Access-Control-Allow-Credentials', 'true')
    if cors.get('allowHeaders'):
        res.headers.add('Access-Control-Allow-Headers', ', '.join(cors.get('allowHeaders')))
    res.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _handle_request(environ):
    env = gws.web.env.prepare(environ)

    req = gws.web.AuthRequest(env)
    if req.params is None:
        raise gws.web.error.NotFound()

    cors = req.site.get('cors')
    if cors and not cors.get('enabled'):
        cors = None

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response)

    req.auth_begin()

    # gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = gws.web.action.handle(req)

    req.auth_commit(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def application(environ, start_response):
    try:
        res = _handle_request(environ)
        return res(environ, start_response)
    except gws.web.error.HTTPException as e:
        return e(environ, start_response)
    except:
        gws.log.exception()
        return gws.web.error.InternalServerError()(environ, start_response)


gws.config.loader.load()

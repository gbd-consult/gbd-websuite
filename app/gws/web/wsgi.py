"""web application root"""

import gws
import gws.common.template
import gws.config.loader
import gws.web.action
import gws.web.auth
import gws.web.error
import gws.web.wrappers

gws.config.loader.load()


def application(environ, start_response):
    res = _handle_request(environ)
    return res(environ, start_response)


##


def _handle_request(environ):
    config_root = gws.config.root()
    environ['gws.site'] = _find_site(environ, config_root)
    req = gws.web.auth.AuthRequest(environ)
    try:
        return _handle_request2(config_root, req)
    except gws.web.error.HTTPException as err:
        return _handle_error(config_root, req, err)
    except:
        gws.log.exception()
        return _handle_error(config_root, req, gws.web.error.InternalServerError())


def _handle_request2(config_root, req):
    if req.params is None:
        raise gws.web.error.NotFound()

    cors = req.site.cors

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response)

    req.auth_begin()

    # gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = gws.web.action.handle(config_root, req)

    req.auth_commit(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def _handle_error(config_root, req, err):
    # @TODO: image errors

    if req.is_json:
        return req.response.content(
            '{"error":{"status":%d,"info":""}}' % err.code,
            'application/json',
            err.code)

    if not req.site.error_page:
        return err

    try:
        context = {
            'request': req,
            'error': err.code
        }
        out = req.site.error_page.render(context)
        return req.response.content(out.content, out.mimeType, err.code)
    except:
        gws.log.exception()
        return gws.web.error.InternalServerError()


def _with_cors_headers(cors, res):
    if cors.get('allowOrigin'):
        res.headers.add('Access-Control-Allow-Origin', cors.get('allowOrigin'))
    if cors.get('allowCredentials'):
        res.headers.add('Access-Control-Allow-Credentials', 'true')
    if cors.get('allowHeaders'):
        res.headers.add('Access-Control-Allow-Headers', ', '.join(cors.get('allowHeaders')))
    res.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _find_site(environ, config_root):
    host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    for s in config_root.application.web_sites:
        if s.host.lower() == host:
            return s
    for s in config_root.application.web_sites:
        if s.host == '*':
            return s

    # there must be a '*' site (see application.config)
    raise ValueError('unknown host', host)

"""web application root"""

import gws
import gws.common.template
import gws.config.loader
import gws.web.auth
import gws.web.error
import gws.web.wrappers

import gws.types as t

gws.config.loader.load()

DEFAULT_CMD = 'assetHttpGetPath'


def application(environ, start_response):
    res = _handle_request(environ)
    return res(environ, start_response)


##


def _handle_request(environ):
    root = gws.config.root()
    environ['gws.site'] = _find_site(environ, root)
    req = gws.web.auth.AuthRequest(environ)
    try:
        return _handle_request2(root, req)
    except gws.web.error.HTTPException as err:
        return _handle_error(root, req, err)
    except:
        gws.log.exception()
        return _handle_error(root, req, gws.web.error.InternalServerError())


def _handle_request2(root, req):
    if req.params is None:
        raise gws.web.error.NotFound()

    cors = req.site.cors

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response)

    req.auth_begin()

    ## gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = _handle_action(root, req)

    req.auth_commit(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def _handle_error(root, req, err):
    # @TODO: image errors

    if req.wants_struct:
        return req.response.struct(
            {'error': {'status': err.code, 'info': ''}},
            status=err.code)

    if not req.site.error_page:
        return err

    try:
        context = {
            'request': req,
            'error': err.code
        }
        out = req.site.error_page.render(context)
        return req.response.raw(out.content, out.mimeType, err.code)
    except:
        gws.log.exception()
        return gws.web.error.InternalServerError()


def _handle_action(root, req):
    cmd = req.param('cmd', DEFAULT_CMD)

    # @TODO: add HEAD
    if req.has_struct:
        category = 'api'
    elif req.is_get:
        category = 'http_get'
    elif req.is_post:
        category = 'http_post'
    else:
        raise gws.web.error.MethodNotAllowed()

    try:
        if category == 'api':
            action_type, method_name, payload = root.validate_action(category, cmd, req.params.get('params'))
        else:
            action_type, method_name, payload = root.validate_action(category, cmd, req.params)
    except gws.config.Error as e:
        gws.log.error('ACTION ERROR', e)
        raise gws.web.error.BadRequest()

    project_uid = payload.get('projectUid')

    ## gws.log.debug(f'DISPATCH a={action_type!r} m={method_name!r} projectUid={project_uid!r}')

    action = root.application.find_action(action_type, project_uid)

    if not action:
        gws.log.error('handler not found', cmd)
        raise gws.web.error.NotFound()

    if not req.user.can_use(action):
        gws.log.error('permission denied', cmd)
        raise gws.web.error.Forbidden()

    # method_name does exist on action (action.validate ensures that)
    r = getattr(action, method_name)(req, payload)

    # now, r is a types.Response object
    if r is None:
        gws.log.error('action not handled', cmd)
        raise gws.web.error.NotFound()

    if isinstance(r, t.HttpResponse):
        return req.response.raw(r.content, r.mimeType, r.get('status', 200))

    return req.response.struct(r)


def _with_cors_headers(cors, res):
    if cors.get('allowOrigin'):
        res.headers.add('Access-Control-Allow-Origin', cors.get('allowOrigin'))
    if cors.get('allowCredentials'):
        res.headers.add('Access-Control-Allow-Credentials', 'true')
    if cors.get('allowHeaders'):
        res.headers.add('Access-Control-Allow-Headers', ', '.join(cors.get('allowHeaders')))
    res.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

    return res


def _find_site(environ, root):
    host = environ.get('HTTP_HOST', '').lower().split(':')[0].strip()

    for s in root.application.web_sites:
        if s.host.lower() == host:
            return s
    for s in root.application.web_sites:
        if s.host == '*':
            return s

    # there must be a '*' site (see application.config)
    raise ValueError('unknown host', host)

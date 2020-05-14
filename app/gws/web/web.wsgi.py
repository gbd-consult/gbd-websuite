"""web application root"""

import gws
import gws.common.template
import gws.config.loader
import gws.core.spec
import gws.web.auth
import gws.web.error
import gws.web.wrappers

import gws.types as t

gws.config.loader.load()


def application(environ, start_response):
    res = t.cast(gws.web.wrappers.BaseResponse, _handle_request(environ))
    return res(environ, start_response)


##

_DEFAULT_CMD = 'assetHttpGetPath'


class _DispatchError(gws.Error):
    pass


def _handle_request(environ) -> t.IResponse:
    root = gws.config.root()
    req = gws.web.auth.Request(root, environ, _find_site(environ, root))
    try:
        return _handle_request2(root, req)
    except gws.web.error.HTTPException as err:
        return _handle_error(root, req, err)
    except:
        gws.log.exception()
        return _handle_error(root, req, gws.web.error.InternalServerError())


def _handle_request2(root, req) -> t.IResponse:
    req.init()

    cors = req.site.cors

    if cors and req.method == 'OPTIONS':
        return _with_cors_headers(cors, req.response('', 'text/plain'))

    req.auth_open()

    ## gws.p('REQUEST', {'user': req.user, 'params': req.params})

    res = _handle_action(root, req)

    req.auth_close(res)

    if cors and req.method == 'POST':
        res = _with_cors_headers(cors, res)

    return res


def _handle_error(root, req, err) -> t.IResponse:
    # @TODO: image errors

    if req.output_struct_type:
        return req.struct_response(
            {'error': {'status': err.code, 'info': gws.get(err, 'description', '')}},
            status=err.code)

    if not req.site.error_page:
        return req.error_response(err)

    try:
        r = req.site.error_page.render({
            'request': req,
            'error': err.code
        })
        return req.response(r.content, r.mime, err.code)
    except:
        gws.log.exception()
        return req.error_response(gws.web.error.InternalServerError())


def _handle_action(root: t.IRootObject, req: t.IRequest) -> t.IResponse:
    cmd = req.param('cmd', _DEFAULT_CMD)

    # @TODO: add HEAD
    if req.input_struct_type:
        category = 'api'
    elif req.method == 'GET':
        category = 'http_get'
    elif req.method == 'POST':
        category = 'http_post'
    else:
        raise gws.web.error.MethodNotAllowed()

    try:
        if category == 'api':
            action_type, method_name, payload = _validate_action(root, category, cmd, req.params.get('params'))
        else:
            action_type, method_name, payload = _validate_action(root, category, cmd, req.params)
    except _DispatchError as e:
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
        return req.response(r.content, r.mime, r.get('status', 200))

    if isinstance(r, t.FileResponse):
        return req.file_response(r.path, r.mime, r.get('status', 200), r.get('attachment_name'))

    return req.struct_response(r)


def _validate_action(root: t.IRootObject, category, cmd, payload):
    cc = root.validator.method_spec(cmd)
    if not cc:
        raise _DispatchError('not found', cmd)

    cat = cc['category']
    if cat == 'http' and category.startswith('http'):
        cat = category
    if category != cat:
        raise _DispatchError('wrong command category', category)

    if cc['arg']:
        try:
            payload = root.validator.read_value(payload, cc['arg'], strict=(cat == 'api'))
        except gws.core.spec.Error as e:
            raise _DispatchError(f"invalid parameter {e.args[2]} ({e.args[0]})") from e

    return cc['action'], cc['name'], payload


def _with_cors_headers(cors, res: t.IResponse) -> t.IResponse:
    if cors.get('allow_origin'):
        res.add_header('Access-Control-Allow-Origin', cors.get('allow_origin'))
    if cors.get('allow_credentials'):
        res.add_header('Access-Control-Allow-Credentials', 'true')
    if cors.get('allow_headers'):
        res.add_header('Access-Control-Allow-Headers', ', '.join(cors.get('allow_headers')))
    res.add_header('Access-Control-Allow-Methods', 'POST, OPTIONS')

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

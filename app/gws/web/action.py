import gws
import gws.config
import gws.types as t

from . import error

DEFAULT_CMD = 'assetHttpGetPath'


def handle(config_root, req):
    cmd = req.param('cmd', DEFAULT_CMD)

    # @TODO: add HEAD
    if req.has_struct:
        category = 'api'
    elif req.is_get:
        category = 'http_get'
    elif req.is_post:
        category = 'http_post'
    else:
        raise error.MethodNotAllowed()

    try:
        action_type, method_name, arg = config_root.validate_action(category, cmd, req.params.get('params'))
    except gws.config.Error as e:
        gws.log.error('ACTION ERROR', e)
        raise error.BadRequest()

    gws.log.debug(f'DISPATCH a={action_type} m={method_name}')

    project_uid = arg.get('projectUid') if arg else req.param('projectUid')
    action = config_root.application.find_action(action_type, project_uid)

    if not action:
        gws.log.error('handler not found', cmd)
        raise error.NotFound()

    if not req.user.can_use(action):
        gws.log.error('permission denied', cmd)
        raise error.Forbidden()

    # method_name does exist on action (action.validate ensures that)
    r = getattr(action, method_name)(req, arg)

    # now, r is a types.Response object
    if r is None:
        gws.log.error('action not handled', cmd)
        raise error.NotFound()

    if isinstance(r, t.HttpResponse):
        return req.response.raw(r.content, r.mimeType, r.get('status', 200))

    return req.response.struct(r)

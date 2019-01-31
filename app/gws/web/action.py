import gws
import gws.config

from . import error

DEFAULT_CMD = 'assetHttpGetPath'


def _action_handler(project_uid, action_type):
    root = gws.config.root()

    if project_uid:
        project = root.find('gws.common.project', project_uid)
        if project:
            action = project.action(action_type)
            if action:
                gws.log.debug(f'action: found project: {project_uid} {action_type}')
                return action
            else:
                gws.log.debug(f'action: PROJECT ACTION NOT FOUND: {project_uid} {action_type}')
        else:
            gws.log.debug(f'action: PROJECT NOT FOUND: {project_uid}')

    app = root.find_first('gws.common.application')
    action = app.action(action_type)
    if action:
        gws.log.debug(f'action: found global: {action_type}')
        return action

    gws.log.debug(f'action: NOT FOUND: {project_uid} {action_type}')


def handle(req):
    cmd = req.param('cmd', DEFAULT_CMD)

    # @TODO: add HEAD
    if req.is_json or req.was_rewritten:
        category = 'api'
    elif req.method == 'GET':
        category = 'get'
    elif req.method == 'POST':
        category = 'post'
    else:
        raise error.MethodNotAllowed()

    try:
        action_type, method_name, arg = gws.config.root().validate_action(category, cmd, req.params.get('params'))
    except gws.config.Error as e:
        gws.log.error('ACTION ERROR', e)
        raise error.BadRequest()

    gws.log.debug(f'DISPATCH a={action_type} m={method_name}')

    project_uid = arg.get('projectUid') if arg else req.param('projectUid')
    action = _action_handler(project_uid, action_type)

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

    if req.is_json:
        return req.response.json(r)

    # not a json request, it must be types.HttpResponse
    return req.response.content(r.content, r.mimeType)

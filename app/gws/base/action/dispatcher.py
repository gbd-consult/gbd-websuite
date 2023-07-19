"""Command dispatcher."""

import gws
import gws.spec

from . import core


def dispatch(
        root: gws.IRoot,
        command_category: str,
        command_name: str,
        params: dict,
        user: gws.IUser = None,
        read_options=None,
):
    desc = root.specs.command_descriptor(command_category, command_name)
    if not desc:
        raise gws.NotFoundError(f'{command_category!r}:{command_name!r} not found')

    try:
        request = root.specs.read(params, desc.tArg, options=read_options)
    except gws.spec.ReadError as exc:
        raise gws.BadRequestError(f'{command_category!r}:{command_name!r} read error: {exc!r}') from exc

    action = _action_object(root, desc, command_category, command_name, request.get('projectUid'), user)
    fn = getattr(action, desc.methodName)
    return fn, request


def _action_object(root, desc, command_category, command_name, project_uid, user):
    if not root.app:
        cls = root.specs.get_class(desc.tOwner)
        return cls()

    if command_category == 'cli':
        cls = root.specs.get_class(desc.tOwner)
        return cls()

    project = None

    if project_uid:
        project = root.app.project(project_uid)
        if not project:
            raise gws.NotFoundError(f'{command_category!r}:{command_name!r} project not found: {project_uid!r}')

    if project and project.actionMgr:
        obj = project.actionMgr.get_action(desc)
        if obj and not user.can_use(obj):
            raise gws.ForbiddenError(f'{command_category!r}:{command_name!r} forbidden in project: {project_uid!r}')
        if obj:
            return obj

    obj = root.app.actionMgr.get_action(desc)
    if obj and not user.can_use(obj):
        raise gws.ForbiddenError(f'{command_category!r}:{command_name!r} forbidden')
    if obj:
        return obj

    raise gws.NotFoundError(f'{command_category!r}:{command_name!r} action not found')

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
        gws.log.debug(f'{command_category!r}:{command_name!r} not found')
        raise core.CommandNotFound(command_category, command_name)

    try:
        request = root.specs.read(params, desc.tArg, options=read_options)
    except gws.spec.ReadError as exc:
        gws.log.debug(f'{command_category!r}:{command_name!r} read error: {exc!r}')
        raise core.BadRequest(command_category, command_name) from exc

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
        project = root.app.get_project(project_uid)
        if not project:
            gws.log.debug(f'{command_category!r}:{command_name!r} project not found {project_uid!r}')
            raise core.CommandNotFound(command_category, command_name, project_uid)

    if project and project.actionMgr:
        obj = project.actionMgr.get_action(desc)
        if obj and not user.can_use(obj):
            gws.log.debug(f'{command_category!r}:{command_name!r} forbidden in project {project_uid!r}')
            raise core.CommandForbidden(command_category, command_name, project_uid)
        if obj:
            return obj

    obj = root.app.actionMgr.get_action(desc)
    if obj and not user.can_use(obj):
        gws.log.debug(f'{command_category!r}:{command_name!r} forbidden')
        raise core.CommandForbidden(command_category, command_name)
    if obj:
        return obj

    gws.log.debug(f'{command_category!r}:{command_name!r} action not found')
    raise core.CommandNotFound(command_category, command_name)

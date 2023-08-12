"""Command dispatcher."""

import gws
import gws.spec
import gws.types as t


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

    if command_category == 'cli' or not root.app:
        cls = root.specs.get_class(desc.tOwner)
        action = cls()
    else:
        action = find(root, desc.owner.extName, user, request.get('projectUid'))

    fn = getattr(action, desc.methodName)
    return fn, request


def find(
        root: gws.IRoot,
        ext_name: str,
        user: gws.IUser,
        project_uid: t.Optional[str] = None
) -> t.Optional[gws.IAction]:
    project = None

    if project_uid:
        project = root.app.project(project_uid)
        if not project:
            raise gws.NotFoundError(f'action {ext_name!r}: project not found: {project_uid!r}')

    if project and project.actionMgr:
        obj = project.actionMgr.get_action(ext_name)
        if obj and not user.can_use(obj):
            raise gws.ForbiddenError(f'action {ext_name!r}: forbidden in project: {project_uid!r}')
        if obj:
            return obj

    obj = root.app.actionMgr.get_action(ext_name)
    if obj and not user.can_use(obj):
        raise gws.ForbiddenError(f'action {ext_name!r}: forbidden')
    if obj:
        return obj

    raise gws.NotFoundError(f'action {ext_name!r}: not found')

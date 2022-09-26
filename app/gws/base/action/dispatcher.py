import gws
import gws.base.web.error
import gws.spec


def dispatch(
        root: gws.IRoot,
        command_category: str,
        command_name: str,
        params: dict,
        user: gws.IUser = None,
        strict_mode=True
):
    def get_object():
        if not root.app:
            cls = root.specs.get_class(desc.tOwner)
            return cls()

        if command_category == 'cli':
            cls = root.specs.get_class(desc.tOwner)
            return cls()

        project = None
        project_uid = request.get('projectUid')

        if project_uid:
            project = root.app.get_project(project_uid)
            if not project:
                gws.log.debug(f'{command_category!r}:{command_name!r} project not found {project_uid!r}')
                raise gws.base.web.error.NotFound()

        if project and project.actionMgr:
            obj = project.actionMgr.find_action(desc.tOwner)
            if obj and not user.can_use(obj):
                gws.log.debug(f'{command_category!r}:{command_name!r} forbidden in project {project_uid!r}')
                raise gws.base.web.error.Forbidden()
            if obj:
                return obj

        obj = root.app.actionMgr.find_action(desc.tOwner)
        if obj and not user.can_use(obj):
            gws.log.debug(f'{command_category!r}:{command_name!r} forbidden')
            raise gws.base.web.error.Forbidden()
        if obj:
            return obj

        gws.log.debug(f'{command_category!r}:{command_name!r} action not found')
        raise gws.base.web.error.NotFound()

    ##

    desc = root.specs.command_descriptor(command_category, command_name)
    if not desc:
        gws.log.debug(f'{command_category!r}:{command_name!r} not found')
        raise gws.base.web.error.NotFound()

    try:
        request = root.specs.read(params, desc.tArg, strict_mode=strict_mode)
    except gws.spec.ReadError as exc:
        gws.log.debug(f'{command_category!r}:{command_name!r} read error: {exc!r}')
        raise gws.base.web.error.BadRequest()

    obj = get_object()
    fn = getattr(obj, desc.methodName)
    return fn, request

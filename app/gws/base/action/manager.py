from typing import Optional
import gws
import gws.spec.runtime


def get_action_for_cli(root: gws.Root, action_name: str, project_uid: Optional[str] = None) -> Optional[gws.Action]:
    """Get an action object by its name and optional project UID.

    If no project UID is provided, it searches for the action in the global scope and then in all projects.
    """

    if project_uid:
        project = root.app.project(project_uid)
        if not project:
            gws.log.error(f'project {project_uid!r} not found')
            return None
        act = root.app.actionMgr.find_action(project, action_name, root.app.authMgr.systemUser)
        if act:
            return act
        gws.log.error(f'action {action_name!r} not found in {project_uid!r}')

    act = root.app.actionMgr.find_action(None, action_name, root.app.authMgr.systemUser)
    if act:
        return act

    for project in root.app.projects:
        act = root.app.actionMgr.find_action(project, action_name, root.app.authMgr.systemUser)
        if act:
            gws.log.info(f'using action {action_name!r} from {project.uid!r}')
            return act

    gws.log.error(f'action {action_name!r} not found')


def parse_cli_request(
    root: gws.Root,
    command_category: gws.CommandCategory,
    command_name: str,
    params: dict,
    read_options=None,
):
    """Parse a CLI request and return the action handler and request object."""

    desc = root.specs.command_descriptor(command_category, command_name)
    if not desc:
        raise gws.NotFoundError(f'{command_category}.{command_name} not found')

    try:
        request = root.specs.read(params, desc.tArg, options=read_options)
    except gws.spec.runtime.ReadError as exc:
        raise gws.BadRequestError(f'read error: {exc!r}') from exc

    cls = gws.u.require(root.specs.get_class(desc.tOwner))
    action = cls()

    fn = getattr(action, desc.methodName)
    return fn, request


class Object(gws.ActionManager):
    def actions_for_project(self, project, user):
        d = {}

        for a in project.actions:
            if user.can_use(a):
                d[a.extType] = a

        for a in self.root.app.actions:
            if a.extType not in d and user.can_use(a):
                d[a.extType] = a

        return list(d.values())

    def prepare_action(self, command_category, command_name, params, user, read_options=None):
        desc = self.root.specs.command_descriptor(command_category, command_name)
        if not desc:
            raise gws.NotFoundError(f'{command_category}.{command_name} not found')

        if desc.extCommandCategory == gws.CommandCategory.raw:
            request = gws.Request(
                projectUid=params.get('projectUid'),
                localeUid=params.get('localeUid'),
            )
        else:
            try:
                request = gws.u.require(self.root.specs.read(params, desc.tArg, options=read_options))
            except gws.spec.runtime.ReadError as exc:
                raise gws.BadRequestError(f'read error: {exc!r}') from exc

        action = None

        project_uid = request.get('projectUid')
        if project_uid:
            project = self.root.app.project(project_uid)
            if not project:
                raise gws.NotFoundError(f'project not found: {project_uid!r}')
            if not user.can_use(project):
                raise gws.ForbiddenError(f'project forbidden: {project_uid!r}')
            action = self._find_by_ext_name(project, desc.owner.extName, user)

        if not action:
            action = self._find_by_ext_name(self.root.app, desc.owner.extName, user)

        if not action:
            raise gws.NotFoundError(f'action {desc.owner.extName!r}: not found')

        fn = getattr(action, desc.methodName)
        return fn, request

    def find_action(self, project, ext_type, user):
        if project:
            a = self._find_by_ext_type(project, ext_type, user)
            if a:
                return a
        return self._find_by_ext_type(self.root.app, ext_type, user)

    # @TODO build indexes for this

    def _find_by_ext_name(self, obj, ext_name: str, user: gws.User):
        for a in obj.actions:
            if a.extName == ext_name:
                if not user.can_use(a):
                    raise gws.ForbiddenError(f'action {ext_name!r}: forbidden in {obj!r}')
                return a

    def _find_by_ext_type(self, obj, ext_type: str, user: gws.User):
        for a in obj.actions:
            if a.extType == ext_type:
                if not user.can_use(a):
                    raise gws.ForbiddenError(f'action {ext_type!r}: forbidden in {obj!r}')
                return a

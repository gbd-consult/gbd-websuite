import gws
import gws.spec


class Object(gws.Node, gws.IActionManager):

    def actions_for_project(self, project, user):
        d = {}

        for a in project.actions:
            if user.can_use(a):
                d[a.extType] = a

        for a in self.root.app.actions:
            if a.extType not in d and user.can_use(a):
                d[a.extType] = a

        return list(d.values())

    def prepare_action(
            self,
            command_category: str,
            command_name: str,
            params: dict,
            user: gws.IUser = None,
            read_options=None,
    ):
        desc = self.root.specs.command_descriptor(command_category, command_name)
        if not desc:
            raise gws.NotFoundError(f'{command_category}.{command_name} not found')

        try:
            request = self.root.specs.read(params, desc.tArg, options=read_options)
        except gws.spec.ReadError as exc:
            raise gws.BadRequestError(f'read error: {exc!r}') from exc

        if command_category == 'cli' or not self.root.app:
            cls = self.root.specs.get_class(desc.tOwner)
            action = cls()
        else:
            action = None

            project_uid = request.get('projectUid')
            if project_uid:
                project = self.root.app.project(project_uid)
                if not project:
                    raise gws.NotFoundError(f'project not found: {project_uid!r}')
                action = self._locate(project, desc.owner.extName, user)

            if not action:
                action = self._locate(self.root.app, desc.owner.extName, user)

        if not action:
            raise gws.NotFoundError(f'action {desc.owner.extName!r}: not found')

        fn = getattr(action, desc.methodName)
        return fn, request

    def locate_action(self, *objects, ext_name, user):
        for obj in objects:
            a = self._locate(obj, ext_name, user)
            if a:
                return a

    def _locate(self, obj, ext_name: str, user: gws.IUser):
        for a in obj.actions:
            if a.extName == ext_name:
                if not user.can_use(a):
                    raise gws.ForbiddenError(f'action {ext_name!r}: forbidden in {obj!r}')
                return a

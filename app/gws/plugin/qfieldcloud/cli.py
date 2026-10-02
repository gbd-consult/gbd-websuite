"""CLI"""

from typing import cast, Optional
import gws
import gws.config
import gws.base.action


from . import action

class PackageRequest(gws.Request):
    projectUid: Optional[str]
    qfcProjectUid: str
    dir: str
    actionName: str = ''

class Object(gws.Node):

    @gws.ext.command.cli('qfieldcloudPackage')
    def invoke(self, p: PackageRequest):
        """Package a QField Cloud project."""

        root = gws.config.load()
        project = None
        if p.projectUid:
            project = root.app.project(p.projectUid)
            if not project:
                gws.log.error(f'project {p.projectUid!r} not found')
                return
        
        act_name = p.actionName or 'qfieldcloud'
        act = cast(action.Object, gws.base.action.get_action_for_cli(root, act_name, p.projectUid))
        if not act:
            return

        own_project = cast(Optional[gws.Project], act.find_closest(gws.ext.object.project))
        if own_project and project and own_project.uid != project.uid:
            gws.log.error(f'action {act_name!r} does not belong to project {p.projectUid!r}')
            return
        project = project or own_project


        sys_user = act.root.app.authMgr.systemUser
        act.create_package_from_cli(p.qfcProjectUid, p.dir, project, sys_user)
        
"""CLI"""

from typing import cast
import gws
import gws.config
import gws.base.action


from . import action

class PackageRequest(gws.Request):
    projectUid: str
    qfcProjectUid: str
    dir: str
    actionName: str = ''

class Object(gws.Node):

    @gws.ext.command.cli('qfieldcloudPackage')
    def invoke(self, p: PackageRequest):
        """Package a QField Cloud project."""

        root = gws.config.load()
        project = root.app.project(p.projectUid)
        if not project:
            gws.log.error(f'project {p.projectUid!r} not found')
            return
        
        act_name = p.actionName or 'qfieldcloud'
        act = cast(action.Object, gws.base.action.get_action_for_cli(root, act_name, p.projectUid))
        if not act:
            return
        
        sys_user = act.root.app.authMgr.systemUser
        act.create_package_from_cli(p.qfcProjectUid, p.dir, project, sys_user)
        
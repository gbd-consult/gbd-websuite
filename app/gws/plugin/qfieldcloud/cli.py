"""CLI"""

from typing import cast
import gws
import gws.config
import gws.base.action
import gws.base.auth
import gws.base.web
import gws.lib.jsonx
import gws.lib.vendor.slon


from . import action, core, packager

class PackageRequest(gws.Request):
    projectUid: str
    qfcProjectUid: str
    dir: str

class Object(gws.Node):

    @gws.ext.command.cli('qfieldcloudPackage')
    def invoke(self, p: PackageRequest):
        """Package a QField Cloud project."""

        root = gws.config.load()
        project = root.app.project(p.projectUid)
        if not project:
            gws.log.error(f'project {p.projectUid!r} not found')
            return
        
        act = cast(action.Object, gws.base.action.get_action_for_cli(root, 'qfieldcloud', p.projectUid))
        if not act:
            return
        
        sys_user = act.root.app.authMgr.systemUser
        act.create_package_from_cli(p.qfcProjectUid, p.dir, project, sys_user)
        
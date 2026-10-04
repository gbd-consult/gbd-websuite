"""Command-line interface for the QField Cloud plugin."""

from typing import cast, Optional
import gws
import gws.config
import gws.base.action


from . import action

class PackageRequest(gws.Request):
    """Parameters of the ``qfieldcloudPackage`` command."""

    projectUid: Optional[str]
    """GWS project uid."""
    qfcProjectUid: str
    """QField project uid."""
    dir: str
    """Directory to write the package into."""
    actionName: str = ''
    """Action type or uid, ``qfieldcloud`` by default."""

class Object(gws.Node):
    """QField Cloud CLI commands.

    Provides the ``qfieldcloudPackage`` command, which creates a QField package
    into a directory.
    """

    @gws.ext.command.cli('qfieldcloudPackage')
    def invoke(self, p: PackageRequest):
        """Package a QField Cloud project into a directory."""

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
        
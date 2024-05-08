"""Command-line GEKOS commands."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config

from . import action

gws.ext.new.cli('gekos')


class CreateIndexParams(gws.CliParams):
    projectUid: Optional[str]
    """project uid"""


class Object(gws.Node):

    @gws.ext.command.cli('gekosIndex')
    def do_index(self, p: CreateIndexParams):
        """Create the GEKOS index."""

        root = gws.config.load()
        project = None

        if p.projectUid:
            project = root.app.project(p.projectUid)
            if not project:
                gws.log.error(f'project {p.projectUid} not found')
                return

        act = cast(
            action.Object,
            root.app.actionMgr.find_action(project, 'gekos', root.app.authMgr.systemUser)
        )
        if not act:
            gws.log.error(f'action "gekos" not found')
            return

        act.index.create()

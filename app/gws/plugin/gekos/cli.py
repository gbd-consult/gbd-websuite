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
        act = cast(action.Object, gws.base.action.find(root, 'gekos', root.app.authMgr.systemUser, p.projectUid))
        act.index.create()

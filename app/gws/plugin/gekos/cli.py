"""Command-line GEKOS commands."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config

from . import action


class CreateIndexParams(gws.CliParams):
    """Parameters for creating the GEKOS index."""
    
    projectUid: Optional[str]
    """Project uid."""


@gws.ext.object.cli('gekos')
class Object(gws.Node):

    @gws.ext.command.cli('gekosIndex')
    def do_index(self, p: CreateIndexParams):
        """Create the GEKOS index."""

        root = gws.config.load()
        act = cast(action.Object, gws.base.action.get_action_for_cli(root, 'gekos', p.projectUid))
        act.idx.create()

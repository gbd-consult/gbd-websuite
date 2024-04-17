"""Command-line ALKIS commands."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.lib.cli as cli
import gws.config

from .data import indexer
from . import action

gws.ext.new.cli('alkis')


class CreateIndexParams(gws.CliParams):
    force: Optional[bool]
    """force indexing"""
    cache: Optional[bool]
    """use object cache"""
    projectUid: Optional[str]
    """project uid"""


class Object(gws.Node):

    @gws.ext.command.cli('alkisIndex')
    def do_index(self, p: CreateIndexParams):
        """Create the ALKIS index."""

        root = gws.config.load()
        project = None

        if p.projectUid:
            project = root.app.project(p.projectUid)
            if not project:
                gws.log.error(f'project {p.projectUid} not found')
                return
        act = cast(
            action.Object,
            root.app.actionMgr.find_action(project, 'alkis', root.app.authMgr.systemUser)
        )
        if not act:
            gws.log.error(f'action "alkis" not found')
            return

        status = act.ix.status()
        if status.complete and not p.force:
            gws.log.info(f'ALKIS index ok')
            return

        indexer.run(act.ix, act.dataSchema, with_force=p.force, with_cache=p.cache)

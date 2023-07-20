"""Command-line ALKIS commands."""

import gws
import gws.config
import gws.types as t

from .data import indexer
from . import action

gws.ext.new.cli('alkis')


class CreateIndexParams(gws.CliParams):
    force: t.Optional[bool]
    """force indexing"""
    cache: t.Optional[bool]
    """use object cache"""
    projectUid: t.Optional[str]
    """project uid"""


class Object(gws.Node):

    @gws.ext.command.cli('alkisIndex')
    def do_index(self, p: CreateIndexParams):
        """Create the ALKIS index."""

        root = gws.config.load()
        if p.projectUid:
            project = t.cast(gws.IProject, root.get(p.projectUid))
            act = t.cast(action.Object, project.actionMgr.find_first(action.Object))
        else:
            act = t.cast(action.Object, root.find_first(action.Object))
        if not act:
            gws.log.error('ALKIS action not found')
            return

        indexer.run(act.index, act.dataSchema, with_force=p.force, with_cache=p.cache)

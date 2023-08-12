"""Command-line ALKIS commands."""

import gws
import gws.base.action
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
        act = t.cast(action.Object, gws.base.action.find(root, 'alkis', root.app.authMgr.systemUser, p.projectUid))
        if act.indexExists and not p.force:
            return
        indexer.run(act.index, act.dataSchema, with_force=p.force, with_cache=p.cache)

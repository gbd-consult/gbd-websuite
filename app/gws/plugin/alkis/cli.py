"""Command-line ALKIS commands."""

import gws
import gws.config
import gws.types as t

from .data import index
from .data import indexer
from .data import norbit6 as reader
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

        rdr = reader.Object(act.index.provider, schema=act.dataSchema)
        if p.force:
            act.index.drop()
        if act.index.exists():
            gws.log.info('ALKIS index ok')
            return
        indexer.run(act.index, rdr, with_cache=p.cache)

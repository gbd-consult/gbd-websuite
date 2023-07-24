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
        act = self._find_action(root, p.projectUid)
        if not act:
            gws.log.error('ALKIS action not found')
            return
        if act.indexExists and not p.force:
            return
        indexer.run(act.index, act.dataSchema, with_force=p.force, with_cache=p.cache)

    def _find_action(self, root, project_uid) -> t.Optional[action.Object]:
        if not project_uid:
            return t.cast(action.Object, root.find_first(action.Object))
        project = t.cast(gws.IProject, root.get(project_uid))
        if not project:
            gws.log.error(f'project {project_uid!r} not found')
            return
        if not project.actionMgr:
            return
        return t.cast(action.Object, project.actionMgr.find_first(action.Object))

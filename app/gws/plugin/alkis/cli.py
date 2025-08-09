"""Command-line ALKIS commands."""

from typing import Optional, cast

import gws
import gws.config
import gws.base.action
import gws.lib.jsonx
import gws.lib.osx
from gws.lib.cli import ProgressIndicator

from .data import types as dt
from .data import exporter, index, indexer
from . import action

gws.ext.new.cli('alkis')


class CreateIndexParams(gws.CliParams):
    projectUid: Optional[str]
    """Project uid."""
    force: bool = False
    """Force indexing."""
    cache: bool = False
    """Use object cache."""


class StatusParams(gws.CliParams):
    projectUid: Optional[str]
    """Project uid."""


class DumpParams(gws.CliParams):
    projectUid: Optional[str]
    """Project uid."""
    fs: str
    """Flurstueck UIDs"""
    path: str
    """Path to save the dump."""


# @TODO options to filter data and configure groups


class ExportParams(gws.CliParams):
    projectUid: Optional[str]
    """Project uid."""
    format: str = 'csv'
    """Export format (csv, geojson)."""
    path: str
    """Path to save the export."""


class Object(gws.Node):
    act: action.Object
    ixStatus: dt.IndexStatus

    def _prepare(self, project_uid):
        root = gws.config.load()
        self.act = cast(action.Object, gws.base.action.get_action_for_cli(root, 'alkis', project_uid))
        if not self.act:
            exit(1)

        self.ixStatus = self.act.ix.status()

    @gws.ext.command.cli('alkisIndex')
    def do_index(self, p: CreateIndexParams):
        """Create the ALKIS index."""

        self._prepare(p.projectUid)
        if self.ixStatus.complete and not p.force:
            gws.log.info(f'ALKIS index ok')
            return

        gws.log.info(f'indexing db={self.act.db.uid} dataSchema={self.act.dataSchema} indexSchema={self.act.indexSchema}')
        indexer.run(self.act.ix, self.act.dataSchema, with_force=p.force, with_cache=p.cache)

    @gws.ext.command.cli('alkisStatus')
    def do_status(self, p: StatusParams):
        """Display the status of the ALKIS index."""

        self._prepare(p.projectUid)

        if self.ixStatus.complete:
            gws.log.info(f'ALKIS index ok')
            return

        if self.ixStatus.missing:
            gws.log.error(f'ALKIS index not found')
            return

        gws.log.warning(
            f'ALKIS index incomplete: basic={self.ixStatus.basic} eigentuemer={self.ixStatus.eigentuemer} buchung={self.ixStatus.buchung}'
        )

    @gws.ext.command.cli('alkisExport')
    def do_export(self, p: ExportParams):
        """Export ALKIS data."""

        self._prepare(p.projectUid)

        if not self.act.exp:
            gws.log.error(f'ALKIS export is not configured')
            exit(3)

        if not self.ixStatus.complete:
            gws.log.error(f'ALKIS index incomplete')
            exit(4)

        if p.format not in {'csv', 'geojson'}:
            gws.log.error(f'invalid format {p.format!r}')
            exit(5)

        qo = dt.FlurstueckQueryOptions(
            withEigentuemer=True,
            withBuchung=True,
            withHistorySearch=False,
            withHistoryDisplay=False,
            displayThemes=[
                'lage',
                'gebaeude',
                'nutzung',
                'festlegung',
                'bewertung',
                'buchung',
                'eigentuemer',
            ],
            limit=1000,
        )

        sys_user = self.act.root.app.authMgr.systemUser
        total = self.act.ix.count_all(qo)
        fs = self.act.ix.iter_all(qo)

        with ProgressIndicator('export', total) as progress:
            path, _ = self.act.exp.run(
                exporter.Args(
                    fs=fs,
                    format=p.format,
                    groups=self.act.exp.groups,
                    user=sys_user,
                    progress=progress,
                )
            )
        gws.lib.osx.rename(path, p.path)

    @gws.ext.command.cli('alkisKeys')
    def do_keys(self, p: gws.CliParams):
        """Print ALKIS export keys."""

        d = index.all_flat_keys()
        for key, typ in d.items():
            print(f'{typ.__name__:7} {key}')

    @gws.ext.command.cli('alkisDump')
    def do_dump(self, p: DumpParams):
        """Dump internal representations of ALKIS objects."""

        self._prepare(p.projectUid)

        if self.ixStatus.missing:
            gws.log.error(f'ALKIS index missing')
            exit(1)

        qo = dt.FlurstueckQueryOptions(
            withEigentuemer=True,
            withBuchung=True,
            withHistorySearch=False,
            withHistoryDisplay=True,
            displayThemes=[
                'lage',
                'gebaeude',
                'nutzung',
                'festlegung',
                'bewertung',
                'buchung',
                'eigentuemer',
            ],
        )

        fs = self.act.ix.load_flurstueck(gws.u.to_list(p.fs), qo)
        js = [index.serialize(f, encode_enum_pairs=False) for f in fs]
        gws.u.write_file(p.path, gws.lib.jsonx.to_pretty_string(js))

"""Command-line cache commands."""

from typing import Optional

import gws
import gws.base.action
import gws.lib.uom
import gws.lib.cli as cli
import gws.config

from . import core

gws.ext.new.cli('cache')


class StatusParams(gws.CliParams):
    layer: Optional[list[str]]
    """list of layer IDs"""


class DropParams(gws.CliParams):
    layer: Optional[list[str]]
    """list of layer IDs"""


class SeedParams(gws.CliParams):
    layer: list[str]
    """list of layer IDs"""
    levels: list[int]
    """zoom levels to build the cache for"""


class Object(gws.Node):

    @gws.ext.command.cli('cacheStatus')
    def do_status(self, p: StatusParams):
        """Display the cache status.

        Lists information about cache entries including layer details and
        cache statistics for each zoom level.

        Args:
            p: Parameters containing optional list of layer IDs to filter by.

        Returns:
            None. Output is printed to the console.
        """

        root = gws.config.loader.load()
        status = core.status(root, gws.u.to_list(p.layer))

        for e in status.entries:
            cli.info('')
            cli.info('=' * 80)

            cli.info(f'CACHE  {e.uid}')
            cli.info(f'DIR    {e.dirname}')

            ls = []
            for la in e.layers:
                title = gws.u.get(la, 'title', '?')
                ls.append(f'{la.uid}: {title!r} type={la.extType}')
            cli.info(f'LAYER  : {_comma(ls)}')

            table = []

            for z, g in sorted(e.grids.items()):
                table.append({
                    'level': z,
                    'scale': '1:' + str(round(gws.lib.uom.res_to_scale(g.res))),
                    'grid': f'{g.maxX} x {g.maxY}',
                    'total': g.totalTiles,
                    'cached': g.cachedTiles,
                    '%%': int(100 * (g.cachedTiles / g.totalTiles)),
                })
            cli.info('')
            cli.info(cli.text_table(table, ['level', 'scale', 'grid', 'total', 'cached', '%%']))

        if status.staleDirs:
            cli.info('')
            cli.info('=' * 80)
            cli.info(f'{len(status.staleDirs)} STALE CACHES ("gws cache cleanup" to remove):')
            for d in status.staleDirs:
                cli.info(f'    {d}')

    @gws.ext.command.cli('cacheCleanup')
    def do_cleanup(self, p: gws.CliParams):
        """Remove stale cache directories.

        Identifies and removes cache directories that are no longer
        associated with any active layers.

        Args:
            p: CLI parameters (unused in this function).

        Returns:
            None. Stale cache directories are removed from the filesystem.
        """

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove active cache directories.

        Removes cache directories for specified layers or all layers if none
        are specified.

        Args:
            p: Parameters containing optional list of layer IDs to drop caches for.

        Returns:
            None. Cache directories are removed from the filesystem.
        """

        root = gws.config.loader.load()
        core.drop(root, gws.u.to_list(p.layer))

    @gws.ext.command.cli('cacheSeed')
    def do_seed(self, p: SeedParams):
        """Seed cache for layers.

        Generates tiles and populates the cache for specified layers and zoom levels.

        Args:
            p: Parameters containing list of layer IDs and zoom levels to generate cache for.

        Returns:
            None. Cache is populated with generated tiles.
        """

        root = gws.config.loader.load()
        status = core.status(root, gws.u.to_list(p.layer))

        levels = []
        if p.levels:
            levels = [int(x) for x in gws.u.to_list(p.levels)]

        core.seed(root, status.entries, levels)


def _comma(items: list) -> str:
    """Join list items with commas.

    Args:
        items: List of items to join.

    Returns:
        String containing comma-separated items.
    """
    return ','.join(items)
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
        """Display the cache status."""

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
        """Remove stale cache dirs."""

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove active cache dirs."""

        root = gws.config.loader.load()
        core.drop(root, gws.u.to_list(p.layer))


_comma = ','.join

#
# _SEED_LOCKFILE = gws.c.CONFIG_DIR + '/mapproxy.seed.lock'
#
#
# @arg('--layers', help='comma separated list of layer IDs')
# @arg('--levels', help='zoom levels to build the cache for')
# def seed(layers=None, levels=None):
#     """Start the cache seeding process"""
#
#     root = gws.config.loader.load()
#
#     if layers:
#         layers = _as_list(layers)
#
#     st = gws.gis.cache.status(root, layers)
#
#     if not st:
#         cli.info('no cached layers found')
#         return
#
#     if levels:
#         levels = [int(x) for x in _as_list(levels)]
#
#     with gws.lib.misc.lock(_SEED_LOCKFILE) as ok:
#         if not ok:
#             cli.info('seed already running')
#             return
#
#         max_time = root.app.cfg('seeding.maxTime')
#         concurrency = root.app.cfg('seeding.concurrency')
#         ts = time.time()
#
#         cli.info(f'\nSTART SEEDING (maxTime={max_time} concurrency={concurrency}), ^C ANYTIME TO CANCEL...\n')
#         done = gws.gis.cache.seed(root, layers, max_time, concurrency, levels)
#         cli.info('=' * 40)
#         cli.info('TIME: %.1f sec' % (time.time() - ts))
#         cli.info(f'SEEDING COMPLETE' if done else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')
#
#

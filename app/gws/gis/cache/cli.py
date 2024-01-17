"""Command-line cache commands."""

import gws
import gws.base.action
import gws.lib.uom
import gws.lib.console
import gws.config
import gws.types as t

from . import core

gws.ext.new.cli('cache')


class StatusParams(gws.CliParams):
    layer: t.Optional[list[str]]
    """list of layer IDs"""


class DropParams(gws.CliParams):
    layer: t.Optional[list[str]]
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
        status = core.status(root, gws.to_list(p.layer))

        for e in status.entries:
            print()
            print('=' * 80)
            print()

            print('CACHE  :', e.uid)
            print('DIR    :', e.dirname)

            ls = []
            for la in e.layers:
                title = gws.get(la, 'title', '?')
                ls.append(f'{la.uid}: {title!r} type={la.extType}')
            print(f'LAYER  :', ', '.join(ls))

            table = []

            for z, g in sorted(e.grids.items()):
                table.append({
                    'zoom': z,
                    'scale': '1:' + str(round(gws.lib.uom.res_to_scale(g.res))),
                    'grid': f'{g.maxX} x {g.maxY}',
                    'total': g.totalTiles,
                    'cached': g.cachedTiles,
                    '%%': int(100 * (g.cachedTiles / g.totalTiles)),
                })
            print()
            print(gws.lib.console.text_table(table, ['zoom', 'scale', 'grid', 'total', 'cached', '%%']))

        if status.staleDirs:
            print()
            print('=' * 80)
            print()
            print(f'{len(status.staleDirs)} STALE CACHES ("gws cache cleanup" to remove):')
            for d in status.staleDirs:
                print(f'    {d}')

        print()

    @gws.ext.command.cli('cacheCleanup')
    def do_cleanup(self, p: gws.CliParams):
        """Remove stale cache dirs."""

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove active cache dirs."""

        root = gws.config.loader.load()
        core.drop(root, gws.to_list(p.layer))

#
# _SEED_LOCKFILE = gws.CONFIG_DIR + '/mapproxy.seed.lock'
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
#         print('no cached layers found')
#         return
#
#     if levels:
#         levels = [int(x) for x in _as_list(levels)]
#
#     with gws.lib.misc.lock(_SEED_LOCKFILE) as ok:
#         if not ok:
#             print('seed already running')
#             return
#
#         max_time = root.app.cfg('seeding.maxTime')
#         concurrency = root.app.cfg('seeding.concurrency')
#         ts = time.time()
#
#         print(f'\nSTART SEEDING (maxTime={max_time} concurrency={concurrency}), ^C ANYTIME TO CANCEL...\n')
#         done = gws.gis.cache.seed(root, layers, max_time, concurrency, levels)
#         print('=' * 40)
#         print('TIME: %.1f sec' % (time.time() - ts))
#         print(f'SEEDING COMPLETE' if done else 'SEEDING INCOMPLETE, PLEASE TRY AGAIN')
#
#

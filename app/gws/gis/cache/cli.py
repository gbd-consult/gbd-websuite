"""Command-line cache commands."""

from typing import Optional

import re

import gws
import gws.lib.jsonx
import gws.lib.cli as cli
import gws.config

from . import core, seed

gws.ext.new.cli('cache')


class FilterParams(gws.CliParams):
    layerUids: Optional[list[str]]
    """List of layer IDs."""
    cacheNames: Optional[list[str]]
    """List of cache names or prefixes."""
    crs: Optional[list[int]]
    """List of CRS codes."""


class StatusParams(FilterParams):
    grids: bool = False
    """Print grid tables."""
    json: str = ''
    """Write json report to path."""


class DropParams(FilterParams):
    pass


class SeedParams(FilterParams):
    levels: str = ''
    """Zoom levels (1,2,3 or 1-3)."""
    maxTime: Optional[int]
    """Max. seeding time in seconds."""
    concurrency: Optional[int]
    """Number of concurrent seeding threads."""
    json: str = ''
    """Write json report to path."""


class Object(gws.Node):
    @gws.ext.command.cli('cacheStatus')
    def do_status(self, p: StatusParams):
        """Display the cache status."""

        root = gws.config.loader.load()
        status = core.status(root, _filter(p))
        js = _status_to_json(status)
        if p.json:
            gws.lib.jsonx.to_path(p.json, js)
        else:
            _display_status(js, p.grids)

    @gws.ext.command.cli('cacheCleanup')
    def do_cleanup(self, p: gws.CliParams):
        """Remove stale cache directories."""

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove active cache directories."""

        root = gws.config.loader.load()
        core.drop(root, _filter(p))

    @gws.ext.command.cli('cacheSeed')
    def do_seed(self, p: SeedParams):
        """Seed cache for layers."""

        root = gws.config.loader.load()

        if p.levels:
            m = re.match(r'^(\d+)-(\d+)$', p.levels)
            if m:
                levels = list(range(int(m.group(1)), int(m.group(2)) + 1))
            else:
                levels = [int(x) for x in gws.u.to_list(p.levels)]
        else:
            levels = []

        opts = core.SeedOptions(
            filter=_filter(p),
            levels=levels,
            maxTime=p.maxTime or root.app.cfg('cache.seedingMaxTime'),
            concurrency=p.concurrency or root.app.cfg('cache.seedingConcurrency'),
        )
        res = seed.seed(root, opts)
        js = _seed_to_json(res)
        if p.json:
            gws.lib.jsonx.to_path(p.json, js)
        else:
            _display_seed(js)


##


def _filter(p: FilterParams) -> core.Filter:
    return core.Filter(
        layerUids=gws.u.to_list(p.layerUids),
        cacheNames=gws.u.to_list(p.cacheNames),
        srids=[int(s) for s in gws.u.to_list(p.crs)],
    )


def _status_to_json(status: core.Status) -> dict:
    entries = []

    for e in status.entries:
        entries.append(
            {
                'name': e.name,
                'dir': e.dir,
                'crs': e.grabber.targetCrs.srid,
                'layers': [
                    {
                        'uid': la.uid,
                        'type': la.extType,
                        'title': gws.u.get(la, 'title', ''),
                    }
                    for la in e.layers
                ],
                'levels': [
                    {
                        'level': lv.z,
                        'gridSize': list(lv.gridSize),
                        'totalTiles': lv.totalTiles,
                        'cachedTiles': lv.cachedTiles,
                        'percentCached': lv.percentCached,
                    }
                    for lv in e.levels
                ],
            }
        )

    return {
        'entries': entries,
        'staleDirs': list(status.staleDirs),
    }


def _display_status(js: dict, with_grids: bool):
    for e in js['entries']:
        cli.info('')
        cli.info('=' * 80)
        cli.info('')

        cli.info(f'CACHE  {e["name"]}')
        # cli.info(f'DIR    {e["dir"] or "-"}')
        # cli.info(f'CRS    {e["crs"] or "-"}')

        la = e['layers'][0]
        uids = ','.join(la['uid'] for la in e['layers'])
        cli.info(f'LAYER  {len(e["layers"])}: {la["type"]} "{la["title"]}" uids={uids}')
        percents = ' '.join(f'{lv["percentCached"]:3d}' for lv in e['levels'])
        cli.info(f'%%     [{percents}]')

        if not with_grids or not e['levels']:
            continue

        table = []

        for lv in e['levels']:
            table.append(
                {
                    'level': lv['level'],
                    'grid': f'{lv["gridSize"][0]} x {lv["gridSize"][1]}',
                    'total': lv['totalTiles'],
                    'cached': lv['cachedTiles'],
                    '%%': lv['percentCached'],
                }
            )
        cli.info('')
        cli.info(cli.text_table(table, ['level', 'grid', 'total', 'cached', '%%']))

    if js['staleDirs']:
        cli.info('')
        cli.info('=' * 80)
        cli.info(f'{len(js["staleDirs"])} STALE CACHES ("gws cache cleanup" to remove):')
        for d in js['staleDirs']:
            cli.info(f'    {d}')


def _seed_to_json(res: core.SeedResult) -> dict:
    entries = []

    for e in res.entries:
        entries.append(
            {
                'name': e.name,
                'dir': e.dir,
                'layers': [
                    {
                        'uid': la.uid,
                        'type': la.extType,
                        'title': gws.u.get(la, 'title', ''),
                    }
                    for la in e.layers
                ],
                'levels': [
                    {
                        'level': lv.z,
                        'totalTiles': lv.totalTiles,
                        'cachedTiles': lv.cachedTiles,
                        'fetchedTiles': lv.fetchedTiles,
                        'failedTiles': lv.failedTiles,
                        'seedTime': round(lv.seedTime, 1),
                    }
                    for lv in e.levels
                ],
                'seedStatus': e.seedStatus or '',
            }
        )

    return {
        'entries': entries,
        'seedTime': round(res.seedTime, 1),
        'seedStatus': res.seedStatus,
    }


def _display_seed(js: dict):
    table = []

    for e in js['entries']:
        for lv in e['levels']:
            if not lv['fetchedTiles'] and not lv['failedTiles']:
                continue
            n = lv['cachedTiles'] + lv['fetchedTiles'] + lv['failedTiles']
            table.append(
                {
                    'cache': e['name'],
                    'level': lv['level'],
                    'total': lv['totalTiles'],
                    'fetched': lv['fetchedTiles'],
                    'failed': lv['failedTiles'],
                    '%%': int(100 * n / lv['totalTiles']) if lv['totalTiles'] else 0,
                    'time': lv["seedTime"],
                    'tps': int(lv["fetchedTiles"] / lv["seedTime"]) if lv["seedTime"] else 0,
                }
            )

    cli.info('')
    cli.info(cli.text_table(table, ['cache', 'level', 'total', 'fetched', 'failed', '%%', 'time', 'tps']))

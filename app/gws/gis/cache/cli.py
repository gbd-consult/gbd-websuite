"""Command-line cache commands."""

from typing import Optional

import re

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.datetimex
import gws.lib.jsonx
import gws.lib.cli as cli
import gws.config

from . import core, seed


class FilterParams(gws.CliParams):
    """Parameters that select caches."""

    layerUids: Optional[list[str]]
    """List of layer IDs."""
    cacheNames: Optional[list[str]]
    """List of cache names or prefixes."""
    crs: Optional[list[int]]
    """List of CRS codes."""


class StatusParams(FilterParams):
    """Parameters of the ``cache status`` command."""

    details: bool = False
    """Print detailed status."""
    json: str = ''
    """Write json report to path."""


class FilterParamsWithGeom(FilterParams):
    """Parameters that select caches, levels and an area."""

    bbox: Optional[list[float]]
    """Bounding box [minx, miny, maxx, maxy]."""
    wkt: str = ''
    """WKT geometry."""
    levels: str = ''
    """Zoom levels (1,2,3 or 1-3)."""


class DropParams(FilterParamsWithGeom):
    """Parameters of the ``cache drop`` command."""

    pass


class SeedParams(FilterParamsWithGeom):
    """Parameters of the ``cache seed`` command."""

    maxTime: Optional[int]
    """Max. seeding time in seconds."""
    concurrency: Optional[int]
    """Number of concurrent seeding threads."""
    maxAge: str = ''
    """Reseed tiles older than this duration (c.g. 1d, 0 to reseed all)."""
    json: str = ''
    """Write json report to path."""


@gws.ext.object.cli('cache')
class Object(gws.Node):
    """Command line commands for the tile cache."""

    @gws.ext.command.cli('cacheStatus')
    def do_status(self, p: StatusParams):
        """Display the cache status."""

        root = gws.config.loader.load()
        inv = core.inventory(root)
        core.apply_filter(inv, _filter(p))
        core.add_stats(inv)
        if p.json:
            gws.lib.jsonx.to_path(p.json, _status_to_json(inv))
        elif p.details:
            _display_status_details(inv)
        else:
            _display_status_brief(inv)

    @gws.ext.command.cli('cacheCleanup')
    def do_cleanup(self, p: gws.CliParams):
        """Remove orphan cache directories."""

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove cached tiles."""

        root = gws.config.loader.load()
        core.drop(root, _filter_with_geom(p))

    @gws.ext.command.cli('cacheSeed')
    def do_seed(self, p: SeedParams):
        """Seed the selected caches."""

        root = gws.config.loader.load()

        opts = core.SeedOptions(
            filter=_filter_with_geom(p),
            maxTime=p.maxTime or root.app.cfg('cache.seedingMaxTime'),
            concurrency=p.concurrency or root.app.cfg('cache.seedingConcurrency'),
            maxAge=gws.lib.datetimex.parse_duration(p.maxAge) if p.maxAge else None,
        )
        res = seed.seed(root, opts)
        if p.json:
            gws.lib.jsonx.to_path(p.json, _seed_to_json(res))
        else:
            _display_seed(res)


##


def _filter(p: FilterParams) -> core.Filter:
    return core.Filter(
        layerUids=gws.u.to_list(p.layerUids),
        cacheNames=gws.u.to_list(p.cacheNames),
        srids=[int(s) for s in gws.u.to_list(p.crs)],
    )


def _levels(s: str) -> list[int]:
    if not s:
        return []
    m = re.match(r'^(\d+)-(\d+)$', s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in gws.u.to_list(s)]


def _filter_with_geom(p: FilterParamsWithGeom) -> core.Filter:
    flt = _filter(p)
    flt.levels = _levels(p.levels)

    bbox = gws.u.to_list(p.bbox)
    wkt = p.wkt or ''
    if not bbox and not wkt:
        return flt

    if bbox and wkt:
        raise gws.Error('only one of bbox and wkt can be given')
    if len(flt.srids) != 1:
        raise gws.Error('exactly one crs is required with bbox or wkt')

    crs = gws.lib.crs.require(flt.srids[0])

    if bbox:
        if len(bbox) != 4:
            raise gws.Error(f'invalid bbox {p.bbox!r}')
        flt.bbox = gws.Bounds(crs=crs, extent=tuple(float(v) for v in bbox))
    else:
        flt.bbox = gws.base.shape.from_wkt(wkt, crs).bounds()

    return flt


def _status_to_json(inv: core.Inventory) -> dict:
    caches = []

    for c in inv.caches:
        caches.append(
            {
                'name': c.name,
                'dir': c.dir,
                'crs': c.grabber.targetCrs.srid,
                'layers': [
                    {
                        'uid': la.uid,
                        'type': la.extType,
                        'title': gws.u.get(la, 'title', ''),
                    }
                    for la in c.layers
                ],
                'levels': [
                    {
                        'level': lv.z,
                        'gridSize': list(lv.gridSize),
                        'gridRange': list(lv.gridRange),
                        'totalTiles': lv.totalTiles,
                        'cachedTiles': lv.cachedTiles,
                        'fileSize': lv.fileSize,
                        'percentCached': lv.percentCached,
                    }
                    for lv in c.levels
                ],
            }
        )

    return {
        'caches': caches,
        'orphanDirs': list(inv.orphanDirs),
    }


def _display_status_brief(inv: core.Inventory):
    cli.info(f'{len(inv.caches)} CACHES')
    if inv.orphanDirs:
        cli.info(f'{len(inv.orphanDirs)} ORPHAN DIRECTORIES')
    cli.info('')

    table = []
    for c in inv.caches:
        ls = c.layerType + ' "' + c.layerTitle[:30] + '"'
        if len(c.layers) > 1:
            ls += f' (+{len(c.layers) - 1})'
        percents = ' '.join(f'{z}:{p}' for z, p in enumerate(core.percentage_by_level(c)))
        table.append(
            {
                'cache': c.name,
                'layer': ls,
                'levels': max(lv.z for lv in c.levels),
                'files': c.cachedTiles,
                'size': _format_file_size(c.fileSize),
                '%%': percents,
            }
        )

    cli.info(cli.text_table(table, header='auto'))


def _display_status_details(inv: core.Inventory):
    cli.info(f'{len(inv.caches)} CACHES')
    if inv.orphanDirs:
        cli.info(f'{len(inv.orphanDirs)} ORPHAN DIRECTORIES')
    cli.info('')

    for c in inv.caches:
        ls = c.layerType + ' "' + c.layerTitle[:30] + '"'
        if len(c.layers) > 1:
            ls += f' (+{len(c.layers) - 1})'
        percents = ' '.join(f'{z}:{p}' for z, p in enumerate(core.percentage_by_level(c)))

        cli.info('')
        cli.info('=' * 80)
        cli.info('')
        cli.info(f'CACHE  {c.name}')
        cli.info(f'LAYER  {ls}')
        cli.info(f'FILES  {c.cachedTiles}, {_format_file_size(c.fileSize)}')
        cli.info(f'%%     {percents}')

        if not c.levels:
            continue

        table = []

        for lv in c.levels:
            table.append(
                {
                    'level': lv.z,
                    'range': f'{lv.gridRange[0]},{lv.gridRange[1]} - {lv.gridRange[2]},{lv.gridRange[3]}',
                    'grid': f'{lv.gridSize[0]} x {lv.gridSize[1]}',
                    'total': lv.totalTiles,
                    'cached': lv.cachedTiles,
                    '%%': lv.percentCached,
                }
            )
        cli.info('')
        cli.info(cli.text_table(table, header='auto'))

    if inv.orphanDirs:
        cli.info('')
        cli.info('=' * 80)
        cli.info(f'{len(inv.orphanDirs)} ORPHAN DIRECTORIES ("gws cache cleanup" to remove):')
        for d in inv.orphanDirs:
            cli.info(f'    {d}')


def _seed_to_json(res: core.SeedResult) -> dict:
    caches = []

    for c in res.caches:
        caches.append(
            {
                'name': c.name,
                'dir': c.dir,
                'layers': [
                    {
                        'uid': la.uid,
                        'type': la.extType,
                        'title': gws.u.get(la, 'title', ''),
                    }
                    for la in c.layers
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
                    for lv in c.levels
                ],
                'seedStatus': c.seedStatus or '',
            }
        )

    return {
        'caches': caches,
        'seedTime': round(res.seedTime, 1),
        'seedStatus': res.seedStatus,
    }


def _display_seed(res: core.SeedResult):
    table = []

    for c in res.caches:
        for lv in c.levels:
            if not lv.fetchedTiles and not lv.failedTiles:
                continue
            table.append(
                {
                    'cache': c.name,
                    'level': lv.z,
                    'total': lv.totalTiles,
                    'fetched': lv.fetchedTiles,
                    'failed': lv.failedTiles,
                    '%%': core.percent_cached(lv),
                    'time': round(lv.seedTime, 1),
                    'tps': int(lv.fetchedTiles / lv.seedTime) if lv.seedTime else 0,
                }
            )

    cli.info('')
    cli.info(cli.text_table(table, header='auto'))


def _format_file_size(size: int) -> str:
    if size == 0:
        return '0'
    s = float(size)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if s < 1024:
            return f'{s:.1f} {unit}'
        s /= 1024
    return str(size) + '??'

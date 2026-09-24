"""Command-line cache commands."""

from typing import Optional

import re

import gws
import gws.base.shape
import gws.lib.crs
import gws.lib.jsonx
import gws.lib.cli as cli
import gws.config

from . import core, seed


class FilterParams(gws.CliParams):
    layerUids: Optional[list[str]]
    """List of layer IDs."""
    cacheNames: Optional[list[str]]
    """List of cache names or prefixes."""
    crs: Optional[list[int]]
    """List of CRS codes."""


class StatusParams(FilterParams):
    details: bool = False
    """Print detailed status."""
    json: str = ''
    """Write json report to path."""


class FilterParamsWithGeom(FilterParams):
    bbox: Optional[list[float]]
    """Bounding box [minx, miny, maxx, maxy]."""
    wkt: str = ''
    """WKT geometry."""


class DropParams(FilterParamsWithGeom):
    levels: str = ''
    """Zoom levels (1,2,3 or 1-3)."""


class SeedParams(FilterParamsWithGeom):
    levels: str = ''
    """Zoom levels (1,2,3 or 1-3)."""
    maxTime: Optional[int]
    """Max. seeding time in seconds."""
    concurrency: Optional[int]
    """Number of concurrent seeding threads."""
    json: str = ''
    """Write json report to path."""


@gws.ext.object.cli('cache')
class Object(gws.Node):
    @gws.ext.command.cli('cacheStatus')
    def do_status(self, p: StatusParams):
        """Display the cache status."""

        root = gws.config.loader.load()
        status = core.apply_filter(core.status(root), _filter(p))
        core.add_counts_and_sizes(status)
        if p.json:
            gws.lib.jsonx.to_path(p.json, _status_to_json(status))
        elif p.details:
            _display_status_details(status)
        else:
            _display_status_brief(status)

    @gws.ext.command.cli('cacheCleanup')
    def do_cleanup(self, p: gws.CliParams):
        """Remove stale cache directories."""

        root = gws.config.loader.load()
        core.cleanup(root)

    @gws.ext.command.cli('cacheDrop')
    def do_drop(self, p: DropParams):
        """Remove active cache directories."""

        root = gws.config.loader.load()
        core.drop(root, _filter_with_geom(p), _levels(p.levels))

    @gws.ext.command.cli('cacheSeed')
    def do_seed(self, p: SeedParams):
        """Seed cache for layers."""

        root = gws.config.loader.load()

        opts = core.SeedOptions(
            filter=_filter_with_geom(p),
            levels=_levels(p.levels),
            maxTime=p.maxTime or root.app.cfg('cache.seedingMaxTime'),
            concurrency=p.concurrency or root.app.cfg('cache.seedingConcurrency'),
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
                        'gridRange': list(lv.gridRange),
                        'totalTiles': lv.totalTiles,
                        'cachedTiles': lv.cachedTiles,
                        'fileSize': lv.fileSize,
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


def _display_status_brief(status: core.Status):
    cli.info(f'{len(status.entries)} CACHES')
    if status.staleDirs:
        cli.info(f'{len(status.staleDirs)} STALE DIRECTORIES')
    cli.info('')

    table = []
    for e in status.entries:
        ls = e.layerType + ' "' + e.layerTitle[:30] + '"'
        if len(e.layers) > 1:
            ls += f' (+{len(e.layers) - 1})'
        percents = ' '.join(f'{z}:{p}' for z, p in enumerate(core.percentage_by_level(e)))
        table.append(
            {
                'cache': e.name,
                'layer': ls,
                'levels': max(lv.z for lv in e.levels),
                'files': e.cachedTiles,
                'size': _format_file_size(e.fileSize),
                '%%': percents,
            }
        )

    cli.info(cli.text_table(table, header='auto'))


def _display_status_details(status: core.Status):
    cli.info(f'{len(status.entries)} CACHES')
    if status.staleDirs:
        cli.info(f'{len(status.staleDirs)} STALE DIRECTORIES')
    cli.info('')

    for e in status.entries:
        ls = e.layerType + ' "' + e.layerTitle[:30] + '"'
        if len(e.layers) > 1:
            ls += f' (+{len(e.layers) - 1})'
        percents = ' '.join(f'{z}:{p}' for z, p in enumerate(core.percentage_by_level(e)))

        cli.info('')
        cli.info('=' * 80)
        cli.info('')
        cli.info(f'CACHE  {e.name}')
        cli.info(f'LAYER  {ls}')
        cli.info(f'FILES  {e.cachedTiles}, {_format_file_size(e.fileSize)}')
        cli.info(f'%%     {percents}')

        if not e.levels:
            continue

        table = []

        for lv in e.levels:
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

    if status.staleDirs:
        cli.info('')
        cli.info('=' * 80)
        cli.info(f'{len(status.staleDirs)} STALE DIRECTORIES ("gws cache cleanup" to remove):')
        for d in status.staleDirs:
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


def _display_seed(res: core.SeedResult):
    table = []

    for e in res.entries:
        for lv in e.levels:
            if not lv.fetchedTiles and not lv.failedTiles:
                continue
            table.append(
                {
                    'cache': e.name,
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

"""Tests for cache management."""

import os

import gws
import gws.gis.cache.core as core
import gws.lib.crs
import gws.lib.grid
import gws.test.util as u

from gws.gis.cache._test import util as tu


@u.fixture(autouse=True)
def system_dirs():
    tu.system_dirs()


def _bbox(mt):
    gr = gws.lib.grid.for_crs(gws.lib.crs.get(3857))
    return gws.Bounds(crs=gws.lib.crs.get(3857), extent=gws.lib.grid.extent_for_tile(gr, mt))


def _level(c, z):
    return next(lv for lv in c.levels if lv.z == z)


def _inventory(*grabbers):
    return core.inventory(tu.Root([tu.layer(f'layer_{n + 1}', [gr]) for n, gr in enumerate(grabbers)]))


def test_inventory_groups_layers_by_cache_name():
    name = tu.cache_name()
    layer_1 = tu.layer('layer_1', [tu.grabber(name)], title='title_1')
    layer_2 = tu.layer('layer_2', [tu.grabber(name)], title='title_2')
    inv = core.inventory(tu.Root([layer_1, layer_2]))

    assert [c.name for c in inv.caches] == [name]
    c = inv.caches[0]
    assert c.layers == [layer_1, layer_2]
    assert (c.layerTitle, c.layerType) == ('title_1', 'type_1')
    assert c.grabber is layer_1.grabbers[3857]


def test_inventory_skips_grabbers_without_cache():
    inv = _inventory(tu.grabber(max_age=0))
    assert inv.caches == []


def test_inventory_sorts_by_type_and_title():
    layer_1 = tu.layer('layer_1', [tu.grabber()], title='b', ext_type='type_2')
    layer_2 = tu.layer('layer_2', [tu.grabber()], title='b', ext_type='type_1')
    layer_3 = tu.layer('layer_3', [tu.grabber()], title='a', ext_type='type_1')
    inv = core.inventory(tu.Root([layer_1, layer_2, layer_3]))
    assert [c.layers[0].uid for c in inv.caches] == ['layer_3', 'layer_2', 'layer_1']


def test_inventory_levels_up_to_max_level():
    gr = tu.grabber(max_level=3)
    c = _inventory(gr).caches[0]

    assert [lv.z for lv in c.levels] == [0, 1, 2, 3]
    lv = _level(c, 2)
    assert lv.gridRange == gr.tile_range_for_level(2) == (0, 0, 3, 3, 2)
    assert lv.gridSize == (4, 4)
    assert lv.totalTiles == 16
    assert lv.resolution == gws.lib.grid.resolution_for_level(gr.grid, 2)


def test_inventory_finds_dirs_and_orphan_dirs():
    gr = tu.grabber()
    gr.store.write((0, 0, 0), b'abc')
    orphan_dir = gws.u.ensure_dir(f'{gws.c.MAP_CACHE_DIR}/{tu.cache_name()}')

    inv = _inventory(gr)

    assert inv.caches[0].dir == gr.store.baseDir
    assert orphan_dir in inv.orphanDirs
    assert gr.store.baseDir not in inv.orphanDirs


def test_apply_filter():
    gr_1 = tu.grabber('cache_a_' + gws.u.random_string(8))
    gr_2 = tu.grabber('cache_b_' + gws.u.random_string(8))
    gr_3 = tu.grabber('cache_b_' + gws.u.random_string(8), srid=4326)

    def names(**kwargs):
        inv = _inventory(gr_1, gr_2, gr_3)
        core.apply_filter(inv, core.Filter(**kwargs))
        return sorted(c.name for c in inv.caches)

    assert names() == sorted([gr_1.cache.name, gr_2.cache.name, gr_3.cache.name])
    assert names(layerUids=['layer_2']) == [gr_2.cache.name]
    assert names(cacheNames=['cache_b_']) == sorted([gr_2.cache.name, gr_3.cache.name])
    assert names(srids=[4326]) == [gr_3.cache.name]
    assert names(cacheNames=['cache_b_'], srids=[3857]) == [gr_2.cache.name]
    assert names(layerUids=['layer_4']) == []


def test_apply_filter_keeps_orphan_dirs():
    inv = core.Inventory(caches=[], orphanDirs=['dir_1'])
    core.apply_filter(inv, core.Filter())
    assert inv.orphanDirs == ['dir_1']


def test_apply_filter_levels():
    inv = _inventory(tu.grabber(max_level=3))
    core.apply_filter(inv, core.Filter(levels=[1, 3, 5]))
    assert [lv.z for lv in inv.caches[0].levels] == [1, 3]
    assert _level(inv.caches[0], 3).gridRange == (0, 0, 7, 7, 3)


def test_apply_filter_bbox():
    inv = _inventory(tu.grabber(max_level=3))
    core.apply_filter(inv, core.Filter(bbox=_bbox((1, 0, 1))))
    c = inv.caches[0]

    assert [lv.z for lv in c.levels] == [0, 1, 2, 3]
    assert _level(c, 0).gridRange == (0, 0, 0, 0, 0)
    assert _level(c, 1).gridRange == (1, 0, 1, 0, 1)
    lv = _level(c, 3)
    assert lv.gridRange == (4, 0, 7, 3, 3)
    assert lv.gridSize == (4, 4)
    assert lv.totalTiles == 16


def test_apply_filter_bbox_and_levels():
    inv = _inventory(tu.grabber(max_level=3))
    core.apply_filter(inv, core.Filter(bbox=_bbox((2, 1, 2)), levels=[2]))
    assert [(lv.z, lv.gridRange) for lv in inv.caches[0].levels] == [(2, (2, 1, 2, 1, 2))]


def test_apply_filter_removes_caches_without_levels():
    inv = _inventory(tu.grabber(max_level=3))
    core.apply_filter(inv, core.Filter(levels=[5]))
    assert inv.caches == []


def test_apply_filter_removes_levels_outside_bbox():
    inv = _inventory(tu.grabber(max_level=3))
    bbox = gws.Bounds(crs=gws.lib.crs.get(3857), extent=(1e9, 1e9, 2e9, 2e9))
    core.apply_filter(inv, core.Filter(bbox=bbox))
    assert inv.caches == []


def test_add_stats():
    gr = tu.grabber(max_level=2)
    gr.store.write((0, 0, 1), b'abc')
    gr.store.write((1, 1, 1), b'defgh')
    gr.store.write((3, 2, 2), b'x')
    inv = _inventory(gr)
    core.add_stats(inv)
    c = inv.caches[0]

    lv = _level(c, 0)
    assert (lv.cachedTiles, lv.fileSize, lv.cachedRange, lv.percentCached) == (0, 0, None, 0)
    lv = _level(c, 1)
    assert (lv.cachedTiles, lv.fileSize, lv.cachedRange, lv.percentCached) == (2, 8, (0, 0, 1, 1, 1), 50)
    lv = _level(c, 2)
    assert (lv.cachedTiles, lv.fileSize, lv.cachedRange, lv.percentCached) == (1, 1, (3, 2, 3, 2, 2), 6)
    assert (c.cachedTiles, c.fileSize) == (3, 9)


def test_add_stats_uses_store_max_age():
    gr = tu.grabber(max_level=0)
    gr.store.write((0, 0, 0), b'abc')
    tu.make_old(gr.store.path((0, 0, 0)), 120)
    gr.store.maxAge = 60
    inv = _inventory(gr)
    core.add_stats(inv)
    assert inv.caches[0].cachedTiles == 0


def test_percent_cached():
    def pc(cached, fetched, total):
        return core.percent_cached(core.Level(cachedTiles=cached, fetchedTiles=fetched, totalTiles=total))

    assert pc(0, 0, 0) == 0
    assert pc(0, 0, 10) == 0
    assert pc(1, 0, 1000) == 1
    assert pc(999, 0, 1000) == 99
    assert pc(500, 500, 1000) == 100
    assert pc(0, 3, 4) == 75
    assert pc(50, 0, 10) == 100


def test_percentage_by_level():
    c = core.Cache(
        grabber=tu.grabber(max_level=4),
        levels=[
            core.Level(z=1, cachedTiles=1, fetchedTiles=1, totalTiles=4),
            core.Level(z=3, cachedTiles=4, fetchedTiles=0, totalTiles=4),
        ],
    )
    assert core.percentage_by_level(c) == [0, 50, 0, 100, 0]


def test_drop_whole_store():
    gr_1 = tu.grabber()
    gr_2 = tu.grabber()
    gr_1.store.write((0, 0, 0), b'abc')
    gr_2.store.write((0, 0, 0), b'abc')
    root = tu.Root([tu.layer('layer_1', [gr_1]), tu.layer('layer_2', [gr_2])])

    core.drop(root, core.Filter(cacheNames=[gr_1.cache.name]))

    assert not os.path.exists(gr_1.store.baseDir)
    assert gr_2.store.has((0, 0, 0))


def test_drop_levels():
    gr = tu.grabber()
    for z in range(4):
        gr.store.write((0, 0, z), b'abc')
    root = tu.Root([tu.layer('layer_1', [gr])])

    core.drop(root, core.Filter(cacheNames=[gr.cache.name], levels=[1, 2]))

    assert [gr.store.has((0, 0, z)) for z in range(4)] == [True, False, False, True]


def test_drop_bbox():
    gr = tu.grabber()
    tiles = [(x, y, 2) for x in range(4) for y in range(4)]
    for mt in tiles:
        gr.store.write(mt, b'abc')
    gr.store.write((0, 0, 1), b'abc')
    root = tu.Root([tu.layer('layer_1', [gr])])

    core.drop(root, core.Filter(cacheNames=[gr.cache.name], bbox=_bbox((1, 0, 1)), levels=[2]))

    left = [mt for mt in tiles if gr.store.has(mt)]
    assert left == [mt for mt in tiles if not (2 <= mt[0] <= 3 and 0 <= mt[1] <= 1)]
    assert gr.store.has((0, 0, 1))


def test_cleanup_removes_orphan_dirs():
    gr = tu.grabber()
    gr.store.write((0, 0, 0), b'abc')
    orphan_dir = gws.u.ensure_dir(f'{gws.c.MAP_CACHE_DIR}/{tu.cache_name()}')

    core.cleanup(tu.Root([tu.layer('layer_1', [gr])]))

    assert not os.path.exists(orphan_dir)
    assert gr.store.has((0, 0, 0))

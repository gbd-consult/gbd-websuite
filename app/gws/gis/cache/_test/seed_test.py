"""Tests for cache seeding."""

import gws
import gws.gis.cache.core as core
import gws.gis.cache.seed as seed
import gws.lib.crs
import gws.lib.grid
import gws.test.util as u

from gws.gis.cache._test import util as tu


@u.fixture(autouse=True)
def system_dirs():
    tu.system_dirs()


def _seed(gr, **kwargs):
    root = tu.Root([tu.layer('layer_1', [gr])])
    opts = core.SeedOptions(filter=core.Filter(cacheNames=[gr.cache.name]), maxTime=60, concurrency=1)
    for k, v in kwargs.items():
        setattr(opts, k, v)
    return seed.seed(root, opts)


def _fetched(res):
    return {lv.z: lv.fetchedTiles for c in res.caches for lv in c.levels}


def _all_tiles(gr, zs):
    return [mt for z in zs for mt in gws.lib.grid.enum_tiles(gr.tile_range_for_level(z))]


def test_seed_fetches_missing_tiles():
    gr = tu.grabber(max_level=3)
    res = _seed(gr)

    assert res.seedStatus == ''
    assert _fetched(res) == {0: 1, 1: 4, 2: 16, 3: 64}
    assert all(gr.store.has(mt) for mt in _all_tiles(gr, range(4)))
    assert len(gr.fetches) == 7


def test_seed_skips_present_tiles():
    gr = tu.grabber(max_level=2)
    _seed(gr)
    gr.fetches = []

    res = _seed(gr)

    assert _fetched(res) == {0: 0, 1: 0, 2: 0}
    assert {lv.z: lv.cachedTiles for lv in res.caches[0].levels} == {0: 1, 1: 4, 2: 16}
    assert gr.fetches == []


def test_seed_levels():
    gr = tu.grabber(max_level=3)
    res = _seed(gr, filter=core.Filter(cacheNames=[gr.cache.name], levels=[2]))

    assert _fetched(res) == {2: 16}
    assert all(gr.store.has(mt) for mt in _all_tiles(gr, [2]))
    assert not any(gr.store.has(mt) for mt in _all_tiles(gr, [0, 1, 3]))


def test_seed_bbox():
    gr = tu.grabber(max_level=3)
    ext = gws.lib.grid.extent_for_tile(gr.grid, (1, 0, 1))
    flt = core.Filter(cacheNames=[gr.cache.name], levels=[3], bbox=gws.Bounds(crs=gws.lib.crs.get(3857), extent=ext))
    res = _seed(gr, filter=flt)

    inside = list(gws.lib.grid.enum_tiles((4, 0, 7, 3, 3)))
    assert _fetched(res) == {3: 16}
    assert all(gr.store.has(mt) for mt in inside)
    assert not any(gr.store.has(mt) for mt in _all_tiles(gr, [3]) if mt not in inside)


def test_seed_max_age_refetches_old_tiles():
    gr = tu.grabber(max_level=1)
    _seed(gr)
    tu.make_old(gr.store.baseDir, 120)
    gr.fetches = []

    res = _seed(gr)
    assert _fetched(res) == {0: 0, 1: 0}

    res = _seed(tu.grabber(gr.cache.name, max_level=1), maxAge=60)
    assert _fetched(res) == {0: 1, 1: 4}
    gr.store.maxAge = 60
    assert all(gr.store.has(mt) for mt in _all_tiles(gr, [0, 1]))


def test_seed_max_age_is_capped_by_cache_max_age():
    gr = tu.grabber(max_age=60, max_level=1)
    _seed(gr)
    tu.make_old(gr.store.baseDir, 120)

    res = _seed(tu.grabber(gr.cache.name, max_age=60, max_level=1), maxAge=3600)

    assert _fetched(res) == {0: 1, 1: 4}


def test_seed_max_age_zero_refetches_all_tiles():
    gr = tu.grabber(max_level=1)
    _seed(gr)

    res = _seed(tu.grabber(gr.cache.name, max_level=1), maxAge=0)

    assert _fetched(res) == {0: 1, 1: 4}


def test_seed_counts_failed_tiles():
    gr = tu.grabber(max_level=1)
    gr.error = ValueError('error_1')
    res = _seed(gr)

    assert _fetched(res) == {0: 0, 1: 0}
    assert {lv.z: lv.failedTiles for lv in res.caches[0].levels} == {0: 1, 1: 4}
    assert not any(gr.store.has(mt) for mt in _all_tiles(gr, [0, 1]))


def test_seed_timeout():
    gr = tu.grabber(max_level=1)
    res = _seed(gr, maxTime=0)

    assert res.seedStatus == 'timeout'
    assert res.caches[0].seedStatus == 'timeout'
    assert gr.fetches == []


def test_seed_is_locked_while_another_seed_runs():
    gr = tu.grabber(max_level=1)
    with gws.u.server_lock('seed', 0):
        res = _seed(gr)

    assert res.seedStatus == 'locked'
    assert res.caches == []
    assert gr.fetches == []


def test_seed_concurrency():
    gr = tu.grabber(max_level=3)
    res = _seed(gr, concurrency=4)

    assert _fetched(res) == {0: 1, 1: 4, 2: 16, 3: 64}
    assert all(gr.store.has(mt) for mt in _all_tiles(gr, range(4)))


def test_blocks_are_aligned_to_request_tiles():
    gr = tu.grabber()
    c = core.Cache(grabber=gr, levels=[core.Level(z=5, gridRange=(2, 3, 9, 5, 5))])
    bg = seed._BlockGenerator(c, c.levels)

    assert list(bg.blocks) == [
        (2, 3, 3, 3, 5),
        (4, 3, 7, 3, 5),
        (8, 3, 9, 3, 5),
        (2, 4, 3, 5, 5),
        (4, 4, 7, 5, 5),
        (8, 4, 9, 5, 5),
    ]


def test_block_queue_is_round_robin():
    cache_1 = core.Cache(name='cache_1', grabber=tu.grabber(), levels=[core.Level(z=3, gridRange=(0, 0, 7, 3, 3))])
    cache_2 = core.Cache(name='cache_2', grabber=tu.grabber(), levels=[core.Level(z=3, gridRange=(0, 0, 7, 3, 3))])
    queue = seed._BlockQueue([cache_1, cache_2])

    names = []
    while True:
        p = queue.next_block()
        if not p:
            break
        names.append(p[0].cache.name)

    assert names == ['cache_1', 'cache_2', 'cache_1', 'cache_2']

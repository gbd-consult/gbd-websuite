"""Tests for the filesystem tile store."""

import os

import gws
import gws.gis.cache.store as store

from gws.gis.cache._test import util as tu


def _store(tmp_path, max_age=3600):
    return store.Object(str(tmp_path / 'store_1'), max_age, 'png')


def _files(st):
    return sorted(p[len(st.baseDir) + 1 :] for p in gws.lib.osx.find_files(st.baseDir))


def test_path_layout(tmp_path):
    st = _store(tmp_path)
    assert st.path((1, 2, 3)) == f'{st.baseDir}/03/0000/0001/0000/0002.png'
    assert st.path((12345, 67890, 17)) == f'{st.baseDir}/17/0001/2345/0006/7890.png'


def test_write_read_has(tmp_path):
    st = _store(tmp_path)
    assert st.read((1, 2, 3)) is None
    assert not st.has((1, 2, 3))
    st.write((1, 2, 3), b'blob_1')
    assert st.read((1, 2, 3)) == b'blob_1'
    assert st.has((1, 2, 3))


def test_expired_tile_is_not_read(tmp_path):
    st = _store(tmp_path, max_age=60)
    st.write((1, 2, 3), b'blob_1')
    tu.make_old(st.path((1, 2, 3)), 120)
    assert st.read((1, 2, 3)) is None
    assert not st.has((1, 2, 3))
    st.maxAge = 180
    assert st.has((1, 2, 3))


def test_stats_for_level(tmp_path):
    st = _store(tmp_path)
    st.write((3, 5, 4), b'abc')
    st.write((10001, 2, 4), b'defgh')
    st.write((0, 0, 5), b'x')

    s = st.stats_for_level(4)
    assert (s.count, s.size, s.range) == (2, 8, (3, 2, 10001, 5, 4))

    s = st.stats_for_level(5)
    assert (s.count, s.size, s.range) == (1, 1, (0, 0, 0, 0, 5))


def test_stats_for_missing_level(tmp_path):
    st = _store(tmp_path)
    s = st.stats_for_level(4)
    assert (s.count, s.size, s.range) == (0, 0, None)


def test_stats_for_whole_store_has_no_range(tmp_path):
    st = _store(tmp_path)
    st.write((3, 5, 4), b'abc')
    st.write((0, 0, 5), b'x')
    s = st.stats()
    assert (s.count, s.size, s.range) == (2, 4, None)


def test_stats_skip_empty_temporary_and_expired_files(tmp_path):
    st = _store(tmp_path, max_age=60)
    st.write((1, 1, 4), b'abc')
    st.write((2, 2, 4), b'abc')
    st.write((3, 3, 4), b'abc')
    tu.make_old(st.path((2, 2, 4)), 120)
    with open(st.path((3, 3, 4)), 'wb'):
        pass
    with open(st.path((1, 1, 4)) + '.random_1.tmp', 'wb') as fp:
        fp.write(b'abc')

    s = st.stats_for_level(4)
    assert (s.count, s.size, s.range) == (1, 3, (1, 1, 1, 1, 4))


def test_drop(tmp_path):
    st = _store(tmp_path)
    st.write((1, 1, 4), b'abc')
    st.drop()
    assert not os.path.exists(st.baseDir)
    st.drop()


def test_drop_level(tmp_path):
    st = _store(tmp_path)
    st.write((1, 1, 4), b'abc')
    st.write((1, 1, 5), b'abc')
    st.drop_level(4)
    st.drop_level(6)
    assert _files(st) == ['05/0000/0001/0000/0001.png']


def test_drop_range_removes_only_tiles_in_range(tmp_path):
    st = _store(tmp_path)
    tiles = [(x, y, 5) for x in (0, 1, 9999, 10000, 10001) for y in (0, 3, 10000)]
    for mt in tiles:
        st.write(mt, b'abc')
    st.write((1, 1, 6), b'abc')

    st.drop_range((1, 0, 10000, 3, 5))

    left = [mt for mt in tiles if os.path.isfile(st.path(mt))]
    assert left == [mt for mt in tiles if not (1 <= mt[0] <= 10000 and 0 <= mt[1] <= 3)]
    assert os.path.isfile(st.path((1, 1, 6)))


def test_drop_range_skips_temporary_files(tmp_path):
    st = _store(tmp_path)
    st.write((1, 1, 5), b'abc')
    tmp = st.path((1, 1, 5)) + '.random_1.tmp'
    with open(tmp, 'wb') as fp:
        fp.write(b'abc')
    st.drop_range((0, 0, 10, 10, 5))
    assert not os.path.exists(st.path((1, 1, 5)))
    assert os.path.isfile(tmp)


def test_drop_range_on_missing_level(tmp_path):
    st = _store(tmp_path)
    st.drop_range((0, 0, 10, 10, 5))
    assert not os.path.exists(st.baseDir)

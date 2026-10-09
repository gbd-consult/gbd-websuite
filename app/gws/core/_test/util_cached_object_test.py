"""Tests for get_cached_object."""

import gws
import gws.test.util as u


@u.fixture(autouse=True)
def object_cache_dir():
    gws.u.ensure_dir(gws.c.OBJECT_CACHE_DIR)


def test_falsy_object_is_cached():
    calls = []

    def init():
        calls.append(1)
        return []

    name = 'test_falsy_object_is_cached_' + gws.u.random_string(8)
    assert gws.u.get_cached_object(name, 100, init) == []
    assert gws.u.get_cached_object(name, 100, init) == []
    assert len(calls) == 1

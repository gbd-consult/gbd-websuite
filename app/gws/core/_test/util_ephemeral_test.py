"""Tests for ephemeral paths."""

import os
import time

import gws
import gws.test.util as u


@u.fixture(autouse=True)
def ephemeral_dir():
    gws.u.ensure_dir(gws.c.EPHEMERAL_DIR)


def _write(path, age):
    gws.u.ensure_dir(os.path.dirname(path))
    gws.u.write_file(path, 'x')
    t = time.time() - age
    os.utime(path, (t, t))


def test_cleanup_removes_old_files_and_empty_dirs():
    max_age = gws.u._ephemeral_state['max_age']
    old = gws.c.EPHEMERAL_DIR + '/dir_1/sub_1/file_1'
    new = gws.c.EPHEMERAL_DIR + '/dir_2/file_2'
    _write(old, max_age + 100)
    _write(new, 10)

    proc = gws.u.ephemeral_cleanup(force=True)
    proc.wait()

    assert not os.path.exists(old)
    assert os.path.exists(new)

    # emptied directories are only removed once they are old themselves
    assert os.path.isdir(gws.c.EPHEMERAL_DIR + '/dir_1/sub_1')
    t = time.time() - max_age - 100
    os.utime(gws.c.EPHEMERAL_DIR + '/dir_1/sub_1', (t, t))
    os.utime(gws.c.EPHEMERAL_DIR + '/dir_1', (t, t))

    proc = gws.u.ephemeral_cleanup(force=True)
    proc.wait()

    assert not os.path.exists(gws.c.EPHEMERAL_DIR + '/dir_1')


def test_cleanup_keeps_fresh_empty_dirs():
    path = gws.u.ensure_dir(gws.c.EPHEMERAL_DIR + '/dir_3')

    proc = gws.u.ephemeral_cleanup(force=True)
    proc.wait()

    assert os.path.isdir(path)


def test_cleanup_is_throttled():
    gws.u._ephemeral_state['next_check'] = 0
    assert gws.u.ephemeral_cleanup() is not None
    assert gws.u.ephemeral_cleanup() is None

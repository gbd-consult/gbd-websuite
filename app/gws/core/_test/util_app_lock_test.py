"""Tests for application locks."""

import threading
import time

import gws
import gws.test.util as u


def test_same_name_gives_same_lock():
    assert gws.u.app_lock('lock_1') is gws.u.app_lock('lock_1')


def test_different_names_give_different_locks():
    assert gws.u.app_lock('lock_2') is not gws.u.app_lock('lock_3')


def test_lock_is_reentrant():
    with gws.u.app_lock('lock_4'):
        with gws.u.app_lock('lock_4'):
            pass


def test_different_names_do_not_block():
    with gws.u.app_lock('lock_5'):
        assert gws.u.app_lock('lock_6').acquire(blocking=False)
        gws.u.app_lock('lock_6').release()


def test_same_name_blocks_other_thread():
    released = threading.Event()
    acquired = threading.Event()

    def worker():
        with gws.u.app_lock('lock_7'):
            acquired.set()
            released.wait()

    t = threading.Thread(target=worker)
    with gws.u.app_lock('lock_7'):
        t.start()
        time.sleep(0.1)
        assert not acquired.is_set()
    acquired.wait(timeout=2)
    assert acquired.is_set()
    released.set()
    t.join()


def test_get_app_global_initializes_once():
    calls = []

    def init_fn():
        calls.append(1)
        return 'value_1'

    assert gws.u.get_app_global('global_1', init_fn) == 'value_1'
    assert gws.u.get_app_global('global_1', init_fn) == 'value_1'
    assert len(calls) == 1
    gws.u.delete_app_global('global_1')

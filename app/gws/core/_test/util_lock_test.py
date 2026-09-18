"""Tests for server locks."""

import multiprocessing
import os
import time

import gws
import gws.test.util as u

_mp = multiprocessing.get_context('fork')


@u.fixture(autouse=True)
def locks_dir():
    gws.u.ensure_dir(gws.c.LOCKS_DIR)


def _hold(uid, queue, seconds):
    with gws.u.server_lock(uid, 0):
        queue.put(os.getpid())
        time.sleep(seconds)


def _hold_and_crash(uid):
    gws.u.server_lock(uid, 0).acquire()
    os._exit(1)


def _start_holder(uid, seconds):
    queue = _mp.Queue()
    proc = _mp.Process(target=_hold, args=(uid, queue, seconds))
    proc.start()
    pid = queue.get(timeout=5)
    return proc, pid


def test_lock_is_acquired_and_released():
    with gws.u.server_lock('lock_1', 0):
        with open(gws.c.LOCKS_DIR + '/lock_1') as fp:
            assert fp.read() == str(os.getpid())
    with gws.u.server_lock('lock_1', 0):
        pass


def test_soft_lock_fails_immediately_when_busy():
    proc, pid = _start_holder('lock_2', 1)
    try:
        t = time.time()
        with u.raises(gws.LockBusyError) as exc:
            with gws.u.server_lock('lock_2', 0):
                pass
        assert time.time() - t < 0.5
        assert str(pid) in str(exc.value)
    finally:
        proc.join()


def test_hard_lock_fails_after_timeout_when_busy():
    proc, _ = _start_holder('lock_3', 2)
    try:
        t = time.time()
        with u.raises(gws.LockBusyError):
            with gws.u.server_lock('lock_3', 0.3):
                pass
        assert 0.3 <= time.time() - t < 1.5
    finally:
        proc.join()


def test_hard_lock_waits_for_release():
    proc, _ = _start_holder('lock_4', 0.5)
    try:
        t = time.time()
        with gws.u.server_lock('lock_4', 5):
            assert time.time() - t < 3
    finally:
        proc.join()


def test_lock_survives_holder_crash():
    proc = _mp.Process(target=_hold_and_crash, args=('lock_5',))
    proc.start()
    proc.join()
    with gws.u.server_lock('lock_5', 0):
        pass


def test_lock_busy_error_is_not_gws_error():
    assert not issubclass(gws.LockBusyError, gws.Error)

"""Tests for the osx module."""
import os
import re
import time

import psutil

import gws
import gws.lib.osx as osx
import gws.test.util as u


def test_getenv():
    assert re.match(r'^/home/[\w\s]+$', osx.getenv('HOME'))


def test_getenv_default():
    assert osx.getenv('FOOBAR', 'default') == 'default'


def test_utime():
    assert int(osx.utime()) == int(time.time())


def test_nowait():
    assert 'nano' not in [i.name() for i in psutil.process_iter()]
    osx.run_nowait('nano')
    assert 'nano' in [i.name() for i in psutil.process_iter()]


def test_run():
    assert osx.run('ls') == (0, b'__init__.py\n__pycache__\n_test.py\n')


def test_unlink(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "foo.txt"
    p.write_text("test")
    osx.unlink(p)
    assert not os.listdir(d)


def test_rename(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "foo.txt"
    p.write_text("test")
    osx.rename(p, d / 'bar.txt')
    p = os.listdir(d)[0]
    assert p == 'bar.txt'
    p = d / p
    assert p.read_text() == 'test'


# uid/ gid does not change as expected
def test_chown(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    p = d / 'foo.txt'
    p.write_text('test')
    osx.chown(p, user=1001, group=1002)
    uid = os.stat(p).st_uid
    gid = os.stat(p).st_gid
    assert uid == gid
    assert uid == 1001
    assert gid == 1002


def test_file_mtime(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "foo.txt"
    p.write_text("test")
    assert int(osx.file_mtime(p)) == int(time.time())


def test_dir_mtime(tmp_path):
    d = tmp_path / 'sub'
    d.mkdir()
    p = d / 'foo.txt'
    p.write_text('test')
    assert int(osx.file_mtime(d)) == int(time.time())


def test_file_age(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "foo.txt"
    p.write_text("test")
    time.sleep(1)
    assert osx.file_age(p) == 1


def test_file_size(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "foo.txt"
    p.write_text("te st")
    assert osx.file_size(p) == 5


def test_file_checksum(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    assert osx.file_checksum(p) == '2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae'


# def test_kill_pids():


# not working
def test_running_pids():
    for i in psutil.process_iter():
        assert i.name() in osx.running_pids().get(i.pid)


def test_process_rss_size():
    assert int(osx.process_rss_size('m')) == 101


# generator obj
def test_find_files(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    d = d / "sub2"
    d.mkdir()
    p = d / "bar2.txt"
    p.write_text("foo")
    d = tmp_path / "sub3"
    d.mkdir()
    p = d / "bar3.txt"
    p.write_text("foo")
    test = []
    for i in osx.find_files(tmp_path):
        test.append(i)
    print(type(osx.find_files(tmp_path)))
    assert str(tmp_path / "sub" / "sub2" / "bar2.txt") in test
    assert str(tmp_path / "sub" / "bar.txt") in test
    assert str(tmp_path / "sub3" / "bar3.txt") in test
    assert len(test) == 3


def test_find_directories(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    d = d / "sub2"
    d.mkdir()
    d = tmp_path / "bus"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    test = []
    for i in osx.find_directories(tmp_path):
        test.append(i)
    assert str(tmp_path / "sub") in test
    assert str(tmp_path / "sub" / "sub2") in test
    assert str(tmp_path / "bus") in test
    assert len(test) == 3


def test_parse_path(tmp_path):
    dirname = osx.parse_path(str(tmp_path)).get('dirname')
    ext = osx.parse_path(str(tmp_path)).get('extension')
    print(dirname)
    fname = osx.parse_path(str(tmp_path)).get('filename')
    name = osx.parse_path(str(tmp_path)).get('name')
    assert re.match(r'^/tmp/pytest-of-\w*/pytest-\d*$', dirname)
    assert re.match(r'', ext)
    assert re.match(r'^test_parse_path\d*$', fname)
    assert re.match(r'test_parse_path\d*', name)


def test_file_name(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    assert osx.file_name(str(p)) == 'bar.txt'


def test_abs_path(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    assert osx.abs_path('bar.txt', '/tmp') == '/tmp/bar.txt'


def test_abs_path_isabs(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    assert osx.abs_path('/bar.txt', '/tmp') == '/bar.txt'


def test_abs_path_nobase(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    with u.raises(ValueError):
        osx.abs_path('bar.txt', '')


def test_abs_web_path(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    path = str(p)[4:]
    name = osx.getenv('HOME').split('/')[-1]
    pattern = r'/tmp/pytest-of-' + re.escape(name) + r'/pytest-\d*/test_abs_web_path0/sub/bar.txt'
    assert re.match(pattern, osx.abs_web_path(path, '/tmp'))


def test_abs_web_path_no_filepath(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    path = str(p)[2:]
    assert not osx.abs_web_path(path, '/tmp')


def test_abs_web_path_inv_filename(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar:txt"
    p.write_text("foo")
    path = str(p)[4:]
    assert not osx.abs_web_path(path, '/tmp')


def test_abs_web_path_inv_dir(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    path = str(p)[4:]
    assert not osx.abs_web_path(':' + path, '/tmp')


def test_rel_path(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "bar.txt"
    p.write_text("foo")
    path = str(p)
    base = str(p)[:-12]
    assert osx.rel_path(path, base) == 'sub/bar.txt'
    assert osx.rel_path(path, base + '/foo/bar') == '../../sub/bar.txt'

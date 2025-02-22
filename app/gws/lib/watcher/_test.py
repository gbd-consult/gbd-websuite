import gws
import gws.test.util as u
import gws.lib.watcher as watcher


def _mkdir(p):
    p.mkdir(parents=True)
    return p


def _touch(p):
    p.touch()
    gws.u.sleep(0.05)
    return p


def _change(p):
    p.write_text("test")
    gws.u.sleep(0.05)
    return p


@u.fixture
def cb():
    class Callback:
        def __init__(self):
            self.changes = []

        def __call__(self, *args):
            self.changes.append(args)

        def changed(self):
            yes = bool(self.changes)
            self.changes = []
            return yes

    return Callback()


def test_simple(tmp_path, cb):
    d1 = _mkdir(tmp_path / '1')
    d2 = _mkdir(tmp_path / '2')

    w = watcher.new(cb)
    w.add_directory(d1)
    w.add_directory(d2)

    w.start()

    _touch(d1 / 'a')
    assert cb.changed()

    _touch(d2 / 'b')
    assert cb.changed()

    w.stop()


def test_patterns(tmp_path, cb):
    d = _mkdir(tmp_path / '1')

    w = watcher.new(cb)
    w.add_directory(d, file_pattern=r'\.txt$')

    w.start()

    _touch(d / 'a.txt')
    assert cb.changed()

    _touch(d / 'a.foo')
    assert not cb.changed()

    w.stop()


def test_exclude(tmp_path, cb):
    d = _mkdir(tmp_path / '1')

    w = watcher.new(cb)
    w.add_directory(d)
    w.exclude('ignore_me')

    w.start()

    a = _touch(d / 'not_ignored')
    assert cb.changed()
    _change(a)
    assert cb.changed()

    a = _touch(d / 'ignore_me')
    assert not cb.changed()
    _change(a)
    assert not cb.changed()


def test_files(tmp_path, cb):
    d = _mkdir(tmp_path / '1')

    w = watcher.new(cb)
    w.add_file(d / 'watch_me_1')
    w.add_file(d / 'watch_me_2')

    w.start()

    a = _touch(d / 'watch_me_1')
    assert cb.changed()
    _change(a)
    assert cb.changed()

    a = _touch(d / 'watch_me_2')
    assert cb.changed()
    _change(a)
    assert cb.changed()

    a = _touch(d / 'ignore_me')
    assert not cb.changed()
    _change(a)
    assert not cb.changed()

    w.stop()


def test_recursive(tmp_path, cb):
    a = _mkdir(tmp_path / 'aaa')
    deep = _mkdir(a / 'bbb' / 'ccc')

    w = watcher.new(cb)
    w.add_directory(a, file_pattern=r'\.txt$', recursive=True)

    w.start()

    _touch(deep / 'a.txt')
    assert cb.changed()

    _touch(deep / 'a.foo')
    assert not cb.changed()

    w.stop()

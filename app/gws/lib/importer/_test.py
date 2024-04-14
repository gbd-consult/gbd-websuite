import sys
import os

import pytest

import gws
import gws.lib.importer as importer


@pytest.fixture
def import_log():
    _sys = {}
    _sys['path'] = list(sys.path)
    _sys['modules'] = dict(sys.modules)
    setattr(gws, 'IMPORT_LOG', [])

    yield getattr(gws, 'IMPORT_LOG')

    sys.path.clear()
    sys.path.extend(_sys['path'])
    sys.modules.clear()
    sys.modules.update(_sys['modules'])
    delattr(gws, 'IMPORT_LOG')


@pytest.fixture(scope='module')
def packages_path(tmpdir_factory):
    base = str(tmpdir_factory.mktemp('importer'))
    files = {
        "/p1/__init__.py":
            "gws.MPORT_LOG.append('p1.init')",
        "/p1/mod.py":
            "gws.MPORT_LOG.append('p1.mod')",
        "/p1/dot.py":
            "gws.MPORT_LOG.append('p1.dot'); from . import mod",
        "/p2/deep/root/base/__init__.py":
            "gws.MPORT_LOG.append('p2.base.init')",
        "/p2/deep/root/base/sub/__init__.py":
            "gws.MPORT_LOG.append('p2.base.sub.init')",
        "/p2/deep/root/base/sub/sub2/__init__.py":
            "gws.MPORT_LOG.append('p2.base.sub.sub2.init')",
        "/p2/deep/root/base/sub/sub2/mod.py":
            "gws.MPORT_LOG.append('p2.base.sub.sub2.mod')",
        "/p3/mod.py":
            "gws.MPORT_LOG.append('p3.mod')",
        "/p4/__init__.py":
            "gws.MPORT_LOG.append('p4.init'); from . import circular",
        "/p4/mod.py":
            "gws.MPORT_LOG.append('p4.mod')",
        "/p4/circular.py":
            "gws.MPORT_LOG.append('p4.circular'); from . import mod",
        "/p5/a/same_name/__init__.py":
            "gws.MPORT_LOG.append('p5.a.same_name.init')",
        "/p5/b/same_name/__init__.py":
            "gws.MPORT_LOG.append('p5.b.same_name.init')",
        "/p0/err.py":
            "syntax error",
    }

    for path, text in files.items():
        path = base + path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gws.u.write_file(path, 'import gws; ' + text)

    yield base


def test_app_module():
    mod = importer.import_from_path(gws.c.APP_DIR + '/gws/lib/importer')
    assert mod.__name__ == 'gws.lib.importer'


def test_app_relative_path():
    mod = importer.import_from_path('gws/lib/importer')
    assert mod.__name__ == 'gws.lib.importer'


def test_import_init(import_log, packages_path):
    importer.import_from_path(packages_path + '/p1')
    assert import_log == ['p1.init']


def test_import_named(import_log, packages_path):
    importer.import_from_path(packages_path + '/p1/mod.py')
    assert import_log == ['p1.init', 'p1.mod']


def test_import_with_dot(import_log, packages_path):
    importer.import_from_path(packages_path + '/p1/dot.py')
    assert import_log == ['p1.init', 'p1.dot', 'p1.mod']


def test_import_deep_init(import_log, packages_path):
    importer.import_from_path(packages_path + '/p2/deep/root/base/sub/sub2')
    assert import_log == ['p2.base.init', 'p2.base.sub.init', 'p2.base.sub.sub2.init']


def test_import_deep_named(import_log, packages_path):
    importer.import_from_path(packages_path + '/p2/deep/root/base/sub/sub2/mod.py')
    assert import_log == ['p2.base.init', 'p2.base.sub.init', 'p2.base.sub.sub2.init', 'p2.base.sub.sub2.mod']


def test_import_no_init(import_log, packages_path):
    importer.import_from_path(packages_path + '/p3/mod.py')
    assert import_log == ['p3.mod']


def test_with_circular_dependency(import_log, packages_path):
    importer.import_from_path(packages_path + '/p4/circular.py')
    assert import_log == ['p4.init', 'p4.circular', 'p4.mod']


def test_no_double_import(import_log, packages_path):
    importer.import_from_path(packages_path + '/p3/mod.py')
    importer.import_from_path(packages_path + '/p3/mod.py')
    assert import_log == ['p3.mod']


def test_err_not_found(import_log, packages_path):
    with pytest.raises(importer.Error):
        importer.import_from_path(packages_path + '/BLAH')


def test_err_import_error(import_log, packages_path):
    with pytest.raises(importer.Error):
        importer.import_from_path(packages_path + '/p0/err.py')


def test_err_same_name(import_log, packages_path):
    importer.import_from_path(packages_path + '/p5/a/same_name')
    with pytest.raises(importer.Error):
        importer.import_from_path(packages_path + '/p5/b/same_name')

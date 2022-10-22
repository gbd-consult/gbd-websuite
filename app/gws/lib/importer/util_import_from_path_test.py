import sys
import textwrap
import os
import contextlib

import gws
import gws.core.util as util
import gws.lib.test as test


def _mkfiles(path, d):
    for k, v in d.items():
        if isinstance(v, dict):
            _mkfiles(path + '/' + k, v)
        else:
            test.write_file(path + '/' + k, textwrap.dedent(v).strip())


@contextlib.contextmanager
def _preserve_sys():
    _sys = {}
    _sys['path'] = list(sys.path)
    _sys['modules'] = dict(sys.modules)

    yield

    sys.path.clear()
    sys.path.extend(_sys['path'])
    sys.modules.clear()
    sys.modules.update(_sys['modules'])


@test.fixture(scope='module')
def dummy_packages():
    _mkfiles(test.TEMP_DIR, {
        'dummy_packages': {
            'gwsdummy1': {
                '__init__.py': "import gws; gws.IMPORT_LOG.append('gwsdummy1.init')",
                'sub': {
                    '__init__.py': "import gws; gws.IMPORT_LOG.append('gwsdummy1.sub.init')",
                    'a.py': "import gws; gws.IMPORT_LOG.append('gwsdummy1.sub.a')",
                    'cross.py': "import gws; gws.IMPORT_LOG.append('gwsdummy1.sub.cross'); from . import a"
                },
            },
            'gwsdummy2': {
                '__init__.py': "import gws; gws.IMPORT_LOG.append('gwsdummy2.init');  from . import circular",
                'a.py': "import gws; gws.IMPORT_LOG.append('gwsdummy2.a')",
                'circular.py': "import gws; gws.IMPORT_LOG.append('gwsdummy2.circular'); from . import a"
            },
            'gwsdummy3': {
                'sub1': {
                    'sub2': {
                        'a.py': "import gws; gws.IMPORT_LOG.append('gwsdummy3.sub1.sub2.a')",
                    }
                }
            },
        }
    })


def test_with_existing_module():
    mod = util.import_from_path(gws.APP_DIR + '/gws/core/data.py')
    assert mod.__name__ == 'gws.core.data'


def test_with_relative_path():
    mod = util.import_from_path('gws/core/data.py')
    assert mod.__name__ == 'gws.core.data'


def test_with_existing_root():
    import_log = gws.IMPORT_LOG = []
    util.import_from_path(os.path.dirname(__file__) + '/gwsdummy.py')
    assert import_log == ['gwsdummy']


def test_with_new_root(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy1/sub/a.py')
        assert import_log == ['gwsdummy1.init', 'gwsdummy1.sub.init', 'gwsdummy1.sub.a']


def test_no_double_import(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy1/sub/a.py')
        assert len(import_log) > 0

        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy1/sub/a.py')
        assert len(import_log) == 0


def test_with_cross_dependency(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy1/sub/cross.py')
        assert import_log == ['gwsdummy1.init', 'gwsdummy1.sub.init', 'gwsdummy1.sub.cross', 'gwsdummy1.sub.a']


def test_with_circular_dependency(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy2/circular.py')
        assert import_log == ['gwsdummy2.init', 'gwsdummy2.circular', 'gwsdummy2.a']


def test_without_init(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy3/sub1/sub2/a.py')
        assert import_log == ['gwsdummy3.sub1.sub2.a']


def test_with_directory(dummy_packages):
    with _preserve_sys():
        import_log = gws.IMPORT_LOG = []
        util.import_from_path(test.TEMP_DIR + '/dummy_packages/gwsdummy2')
        assert import_log == ['gwsdummy2.init', 'gwsdummy2.circular', 'gwsdummy2.a']

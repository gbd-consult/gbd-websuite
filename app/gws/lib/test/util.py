"""Test utilities."""

import inspect
import io
import os
import shutil
import time
import re
import configparser

import pytest

import gws
import gws.base.feature
import gws.lib.vendor.slon
import gws.spec.runtime
import gws.base.web.wsgi_app
import gws.base.auth
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.net
import gws.lib.sa as sa
import gws.lib.password

APP_DIR = os.path.realpath(os.path.dirname(__file__) + '/../../../')
TEST_FILE_RE = r'_test\.py$'
TEST_FILE_GLOB = '*_test.py'

fixture = pytest.fixture

_OPTIONS = {}


def option(key, default=None):
    opts = options()
    if key in opts:
        return opts[key]
    if default is not None:
        return default
    raise KeyError(f'test: option {key!r} not found')


def options():
    if _OPTIONS:
        return _OPTIONS

    _OPTIONS.update(read_ini(
        APP_DIR + '/test.ini',
        APP_DIR + '/test.local.ini'
    ))

    env = {}
    for k, v in _OPTIONS.items():
        sec, _, name = k.partition('.')
        if sec == 'environment':
            if v.startswith('./'):
                v = work_dir() + v[1:]
            env[name] = v
    _OPTIONS['environment'] = env

    for k, v in env.items():
        os.environ[k] = v

    return _OPTIONS


def work_dir():
    return option('runner.work_dir')


##

def read_ini(*paths):
    opts = {}
    cc = configparser.ConfigParser()
    cc.optionxform = str

    for path in paths:
        if os.path.isfile(path):
            cc.read(path)

    for sec in cc.sections():
        for opt in cc.options(sec):
            opts[sec + '.' + opt] = cc.get(sec, opt)

    return opts


def make_ini(opts):
    cc = configparser.ConfigParser()

    for k, v in opts.items():
        sec, _, name = k.partition('.')
        if not cc.has_section(sec):
            cc.add_section(sec)
        cc.set(sec, name, v)

    with io.StringIO() as fp:
        cc.write(fp, space_around_delimiters=False)
        return fp.getvalue()


##

_SPEC = None
_CONFIG_DEFAULTS = '''
    database.providers+ { type "postgres" serviceName "gws_test_postgres" } 
'''


def gws_root():
    global _SPEC
    if _SPEC is None:
        _SPEC = gws.spec.runtime.create(work_dir() + '/MANIFEST.json', read_cache=True, write_cache=True)
    return gws.create_root_object(_SPEC)


def gws_configure(config: str):
    wd = work_dir()

    config = _CONFIG_DEFAULTS + '\n' + config
    dct = gws.lib.vendor.slon.parse(config, as_object=True)

    gws.lib.jsonx.to_path(f'{wd}/config.json', dct, pretty=True)
    root = gws.config.configure(
        manifest_path=f'{wd}/MANIFEST.json',
        config_path=f'{wd}/config.json',
        with_spec_cache=False
    )
    gws.config.activate(root)
    return root


def gws_system_user():
    return gws.base.auth.user.SystemUser(None, roles=[])


def gws_model_context(**kwargs):
    kwargs.setdefault('op', gws.ModelOperation.read)
    kwargs.setdefault('user', gws_system_user())
    return gws.ModelContext(**kwargs)


##

def enumerate_files_for_test(only):
    # enumerate files to test, wrt --only

    files = list(gws.lib.osx.find_files(f'{APP_DIR}/gws', TEST_FILE_RE))
    if only:
        files = [f for f in files if re.search(only, f)]

    # sort files semantically

    _sort_order = ['/core/', '/lib/', '/base/', '/plugin/']

    def _sort_key(path):
        for n, s in enumerate(_sort_order):
            if s in path:
                return n, path
        return 99, path

    files.sort(key=_sort_key)
    return files


##

_PG_ENGINE = None
_PG_CONN = None


def pg_connect():
    # kwargs.setdefault('pool_pre_ping', True)
    # kwargs.setdefault('echo', self.root.app.developer_option('db.engine_echo'))

    global _PG_ENGINE, _PG_CONN

    if _PG_ENGINE is None:
        params = {
            'application_name': 'gws_test',
            'service': option('service.postgres.name'),
        }
        url = gws.lib.net.make_url(
            scheme='postgresql',
            hostname='',
            path='',
            params=params,
        )
        _PG_ENGINE = sa.create_engine(url)

    if _PG_CONN is None:
        _PG_CONN = _PG_ENGINE.connect()

    return _PG_CONN


def pg_create(table_name, col_defs):
    conn = pg_connect()
    conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name} CASCADE'))
    ddl = _comma(f'{k} {v}' for k, v in col_defs.items())
    conn.execute(sa.text(f'CREATE TABLE {table_name} ( {ddl} )'))
    conn.commit()


def pg_insert(table_name, row_dicts):
    conn = pg_connect()
    conn.execute(sa.text(f'TRUNCATE TABLE {table_name}'))
    if row_dicts:
        names = _comma(k for k in row_dicts[0])
        values = _comma(':' + k for k in row_dicts[0])
        ins = sa.text(f'INSERT INTO {table_name} ( {names} ) VALUES( {values} )')
        conn.execute(ins, row_dicts)
    conn.commit()


def pg_rows(sql: str) -> list[tuple]:
    conn = pg_connect()
    return [tuple(r) for r in conn.execute(sa.text(sql))]


def pg_exec(sql: str, **kwargs):
    conn = pg_connect()
    for s in sql.split(';'):
        if s.strip():
            conn.execute(sa.text(s.strip()), kwargs)
    conn.commit()


##

def model(root, name) -> gws.IModel:
    return root.get(name)


def db_model(root, name) -> gws.IDatabaseModel:
    return root.get(name)


def feature(model, **atts) -> gws.IFeature:
    f = gws.base.feature.new(model=model, record = gws.FeatureRecord(attributes=atts))
    f.attributes = atts
    return f


##


def setup():
    pass


def teardown():
    gws.lib.osx.unlink(glob.SESSION_STORE_PATH)
    gws.base.web.wsgi_app.reload()
    gws.config.deactivate()
    mockserv.command('reset')


def make_users_json(lst):
    path = '/tmp/gws_test_users.json'
    if lst is None:
        gws.lib.osx.unlink(path)
        return None

    for v in lst:
        v['password'] = gws.lib.password.encode(v['password'])
    gws.lib.jsonx.to_path(path, lst)
    return path


def write_file(path, text):
    pp = gws.lib.osx.parse_path(path)
    if pp['dirname'].startswith(gws.TMP_DIR):
        gws.ensure_dir(pp['dirname'])
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file_if_changed(path, text):
    curr = read_file(path)
    if text != curr:
        write_file(path, text)


def copy_file(path, dir):
    shutil.copy(path, dir)


def rel_path(path):
    f = inspect.stack(2)[1].filename
    return os.path.join(os.path.dirname(f), path)


def sleep(n):
    time.sleep(n)


def dict_of(x):
    if gws.is_data_object(x):
        # noinspection PyTypeChecker
        return dict(sorted(vars(x).items()))
    return x


def fxml(s):
    s = s.strip()
    s = re.sub(r'>\s+<', '><', s)
    s = re.sub(r'\s*>\s*', '>', s)
    s = re.sub(r'\s*<\s*', '<', s)
    s = re.sub(r'\s+', ' ', s)
    return s


_comma = ','.join

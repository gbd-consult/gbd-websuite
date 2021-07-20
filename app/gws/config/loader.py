import os
import pickle

import gws
import gws.types as t
import gws.spec.generator
import gws.spec.runtime
import gws.server

from . import parser, error

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

DEFAULT_STORE_PATH = gws.CONFIG_DIR + '/config.pickle'

DEFAULT_ROOT_NAME = 'GWS_ROOT'


def parse(manifest_path=None, config_path=None, with_spec_cache=True):
    if manifest_path:
        gws.log.info(f'using manifest {manifest_path!r}...')
    specs = gws.spec.runtime.create(manifest_path, with_cache=with_spec_cache)
    config_path = real_config_path(config_path)
    gws.log.info(f'using config {config_path!r}...')
    parsed_config = parser.parse_main(specs, config_path)
    return [specs, parsed_config]


def fallback_config():
    cfg = {
        'server': gws.server.FALLBACK_CONFIG
    }
    specs = gws.spec.runtime.create(manifest_path=None, with_cache=False)
    parsed_config = specs.read_value(cfg, 'gws.base.application.Config', '', strict=True, with_error_details=True)
    return [specs, parsed_config]


def initialize(parse_result) -> gws.RootObject:
    [specs, parsed_config] = parse_result

    r = gws.RootObject()
    r.specs = specs

    try:
        ts = gws.time_start('loading application')
        mod = gws.import_from_path(gws.APP_DIR + '/gws/base/application/__init__.py', 'gws.base.application')
        app = getattr(mod, 'Object')
        gws.time_end(ts)
    except Exception as e:
        raise error.ConfigurationError(*e.args)

    try:
        ts = gws.time_start('configuring application')
        r.create_application(app, parsed_config)
        gws.time_end(ts)
    except Exception as e:
        raise error.ConfigurationError(*e.args)

    try:
        r.post_initialize()
    except Exception as e:
        raise error.ConfigurationError(*e.args)

    return r


def activate(r: gws.RootObject):
    return gws.set_global(DEFAULT_ROOT_NAME, r)


def store(r: gws.RootObject, path=None):
    path = path or DEFAULT_STORE_PATH
    try:
        gws.write_file_b(path, pickle.dumps(r))
    except Exception as e:
        raise error.LoadError('unable to store configuration') from e


def load(path=None) -> gws.RootObject:
    path = path or DEFAULT_STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        r = pickle.loads(gws.read_file_b(path))
        return gws.set_global(DEFAULT_ROOT_NAME, r)
    except Exception as e:
        raise error.LoadError('unable to load configuration') from e


def root() -> gws.RootObject:
    def _err():
        raise error.LoadError('no configuration root found')

    return gws.get_global(DEFAULT_ROOT_NAME, _err)


def real_config_path(config_path=None):
    p = config_path or os.getenv('GWS_CONFIG')
    if p:
        return p
    for p in DEFAULT_CONFIG_PATHS:
        if os.path.exists(p):
            return p

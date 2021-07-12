import os
import pickle

import gws
import gws.types as t
import gws.spec.generator
import gws.spec.runtime
import gws.base.application

from . import parser, error

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

DEFAULT_STORE_PATH = gws.CONFIG_DIR + '/config.pickle'

DEFAULT_ROOT_NAME = 'GWS_ROOT'


def parse(manifest_path=None, config_path=None):
    if manifest_path:
        gws.log.info(f'using manifest {manifest_path!r}...')
    opts = {
        'manifest_path': manifest_path,
        'cache_path': gws.OBJECT_CACHE_DIR + '/server.spec.json',
    }
    specs = gws.spec.runtime.create(opts)
    config_path = real_config_path(config_path)
    gws.log.info(f'using config {config_path!r}...')
    parsed_config = parser.parse_main(specs, config_path)
    parsed_config.specs = specs
    return parsed_config


def initialize(parsed_config) -> gws.RootObject:
    r = gws.RootObject()
    r.specs = parsed_config.specs
    try:
        r.create_application(gws.base.application.Object, parsed_config)
    except Exception as e:
        raise error.ConfigError(*e.args)
    try:
        r.post_initialize()
    except Exception as e:
        raise error.LoadError(*e.args)
    # for p in set(cfg.configPaths):
    #     root.application.monitor.add_path(p)
    # for p in set(cfg.projectPaths):
    #     root.application.monitor.add_path(p)
    # for d in set(cfg.projectDirs):
    #     root.application.monitor.add_directory(d, parser.config_path_pattern)
    #
    # if root.application.developer_option('server.auto_reload'):
    #     root.application.monitor.add_directory(gws.APP_DIR, '\.py$')
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

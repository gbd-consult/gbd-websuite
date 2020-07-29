import os
import pickle

import gws

import gws.types as t

from . import parser, error, gwsroot

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

DEFAULT_STORE_PATH = gws.CONFIG_DIR + '/config.pickle'


def real_config_path(config_path=None):
    p = config_path or os.getenv('GWS_CONFIG')
    if p:
        return p
    for p in DEFAULT_CONFIG_PATHS:
        if os.path.exists(p):
            return p


def parse_and_activate(path=None) -> t.IRootObject:
    path = real_config_path(path)
    gws.log.info(f'using config "{path}"...')
    cfg, cfg_paths = parser.parse_main(path)
    root = activate(cfg)

    for p in set(cfg_paths):
        root.application.monitor.add_path(p)
    for p in cfg.projectPaths:
        root.application.monitor.add_path(p)
    for d in cfg.projectDirs:
        root.application.monitor.add_directory(d, parser.config_path_pattern)

    if root.application.developer_option('server.auto_reload'):
        root.application.monitor.add_directory(gws.APP_DIR, '\.py$')

    return root


def activate(cfg) -> t.IRootObject:
    try:
        root = gwsroot.create()
        root.initialize(cfg)
        root.post_initialize()
        return root
    except error.ParseError:
        raise
    except Exception as e:
        raise error.LoadError(*e.args)


def store(root: t.IRootObject, path=None):
    path = path or DEFAULT_STORE_PATH
    try:
        gws.write_file_b(path, pickle.dumps(root))
    except Exception as e:
        raise error.LoadError('unable to store configuration') from e


def load(path=None) -> t.IRootObject:
    path = path or DEFAULT_STORE_PATH
    try:
        gws.log.debug(f'loading config from "{path}"')
        with open(path, 'rb') as fp:
            r = pickle.load(fp)
        return gws.set_global('_tree_root', r)
    except Exception as e:
        raise error.LoadError('unable to load configuration') from e

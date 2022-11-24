import sys
import json
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


def parse(path=None):
    path = real_config_path(path)
    gws.log.info(f'using config "{path}"...')
    return parser.parse_main(path)


def parse_and_activate(path=None) -> t.IRootObject:
    cfg = parse(path)
    root = activate(cfg)
    return root


def activate(cfg) -> t.IRootObject:
    try:
        root = gwsroot.create()
        root.initialize(cfg)
        root.post_initialize()
    except error.ParseError:
        raise
    except Exception as e:
        raise error.LoadError(*e.args)

    for p in set(cfg.configPaths):
        root.application.monitor.add_path(p)
    for p in set(cfg.projectPaths):
        root.application.monitor.add_path(p)
    for d in set(cfg.projectDirs):
        root.application.monitor.add_directory(d, parser.config_path_pattern)

    if root.application.developer_option('server.auto_reload'):
        root.application.monitor.add_directory(gws.APP_DIR, '\.py$')

    return root


def store(root: t.IRootObject, path=None):
    path = path or DEFAULT_STORE_PATH
    try:
        gws.write_file(path + '.syspath.json', json.dumps(sys.path))
        gws.write_file_b(path, pickle.dumps(root))
    except Exception as e:
        raise error.LoadError('unable to store configuration') from e


def load(path=None) -> t.IRootObject:
    path = path or DEFAULT_STORE_PATH
    try:
        gws.log.debug(f'loading config from "{path}"')
        sys.path = json.loads(gws.read_file(path + '.syspath.json'))
        r = pickle.loads(gws.read_file_b(path))
        gws.set_global('_tree_root', r)
        r.activate()
        return r
    except Exception as e:
        raise error.LoadError('unable to load configuration') from e

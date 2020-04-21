import os
import pickle

import gws
import gws.server.monitor

import gws.types as t

from . import parser, error, globals, gwsroot

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
    cfg = parser.parse_main(path)
    root = activate(cfg)

    project_pattern = f'({parser.config_path_pattern})|(\\.qgs$)'

    gws.server.monitor.add_path(path)

    for p in cfg.get('projectPaths') or []:
        gws.server.monitor.add_directory(p, project_pattern)
    for d in cfg.get('projectDirs') or []:
        gws.server.monitor.add_directory(d, project_pattern)

    return root


def activate(cfg) -> t.IRootObject:
    try:
        root = gws.set_global('_tree_root', gwsroot.Object())
        root.initialize(cfg)
        root.post_initialize()
        return root
    except error.ParseError:
        raise
    except Exception as e:
        raise error.LoadError(*e.args)


def store(path=None):
    path = path or DEFAULT_STORE_PATH
    try:
        r = globals.root()
        with open(path, 'wb') as fp:
            pickle.dump(r, fp)
    except Exception:
        raise error.LoadError('unable to store configuration')
    try:
        os.chown(path, gws.UID, gws.GID)
    except:
        pass


def load(path=None) -> t.IRootObject:
    path = path or DEFAULT_STORE_PATH
    try:
        gws.log.debug(f'loading config from "{path}"')
        with open(path, 'rb') as fp:
            r = pickle.load(fp)
        return gws.set_global('_tree_root', r)
    except Exception:
        raise error.LoadError('unable to load configuration')

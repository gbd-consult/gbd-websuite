import os
import pickle

import gws
import gws.server.monitor
from . import parser, error, globals, gwsroot

DEFAULT_CONFIG_PATHS = [
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

DEFAULT_STORE_PATH = gws.CONFIG_DIR + '/config.pickle'


def real_config_path(config_path=None):
    p = config_path or os.environ.get('GWS_CONFIG')
    if p:
        return p
    for p in DEFAULT_CONFIG_PATHS:
        if os.path.exists(p):
            return p


def parse_and_activate(path=None):
    path = real_config_path(path)
    gws.log.info(f'using config "{path}"...')
    cfg = parser.parse_main(path)
    activate(cfg)

    project_pattern = f'({parser.config_path_pattern})|(\\.qgs$)'

    gws.server.monitor.add_path(path)

    for p in cfg.get('projectPaths') or []:
        gws.server.monitor.add_directory(p, project_pattern)
    for d in cfg.get('projectDirs') or []:
        gws.server.monitor.add_directory(d, project_pattern)

    return cfg


def activate(cfg):
    try:
        # @TODO get rid of the global root altogether
        r = gws.set_global('_tree_root', gwsroot.Object())
        r.initialize(cfg)
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


def load(path=None):
    path = path or DEFAULT_STORE_PATH
    try:
        gws.log.info(f'loading config from "{path}"')
        with open(path, 'rb') as fp:
            r = pickle.load(fp)
        gws.set_global('_tree_root', r)
    except Exception:
        raise error.LoadError('unable to load configuration')

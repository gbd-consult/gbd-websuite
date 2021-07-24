import os

import gws
import gws.spec.runtime
from . import error, parser

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

_STORE_PATH = gws.TMP_DIR + '/config.pickle'
_ROOT_NAME = 'gws_root_object'


def configure_server(manifest_path=None, config_path=None, config=None,
                     before_init=None, fallback_config=None, with_spec_cache=True) -> gws.RootObject:
    """Configure the server"""

    with_fallback = bool(fallback_config)

    def _print(a):
        if isinstance(a, (list, tuple)):
            for item in a:
                _print(item)
        elif a is not None:
            for s in gws.lines(str(a)):
                gws.log.error(s)

    def _report(exc):
        if isinstance(exc, gws.config.ParseError):
            _print('CONFIGURATION PARSE ERROR:')
        elif isinstance(exc, gws.config.ConfigurationError):
            _print('CONFIGURATION ERROR:')
        elif isinstance(exc, gws.config.LoadError):
            _print('CONFIGURATION LOAD ERROR:')
        _print(exc.args)

    def _handle(exc):
        if not with_fallback:
            raise gws.Error('configuration failed') from exc
        gws.log.warn(f'configuration error: using fallback config')
        return configure_server(config=fallback_config, with_spec_cache=False)

    if manifest_path:
        gws.log.info(f'using manifest {manifest_path!r}...')

    try:
        specs = gws.spec.runtime.create(manifest_path, with_cache=with_spec_cache)
    except Exception as exc:
        _report(exc)
        return _handle(exc)

    with_fallback = with_fallback and specs.manifest.withFallbackConfig

    if not config:
        try:
            config = parse(specs, config_path)
        except Exception as exc:
            _report(exc)
            return _handle(exc)

    if before_init:
        try:
            before_init(config)
        except Exception as exc:
            _report(exc)
            return _handle(exc)

    try:
        root = initialize(specs, config)
    except Exception as exc:
        _report(exc)
        return _handle(exc)

    if not root.configuration_errors:
        gws.log.info(f'configuration ok')
        return root

    if not specs.manifest.withStrictConfig:
        n = len(root.configuration_errors)
        gws.log.error(f'{n} CONFIGURATION ERRORS')
        return root

    args = []
    for err, stk in root.configuration_errors:
        args.append('--------------------------------')
        args.append(err)
        args.extend(stk)

    try:
        raise error.ConfigurationError(*args)
    except Exception as exc:
        _report(exc)
        return _handle(exc)


def parse(specs: gws.ISpecRuntime, config_path=None):
    config_path = real_config_path(config_path)
    gws.log.info(f'using config {config_path!r}...')
    parsed_config = parser.parse_main(specs, config_path)
    return parsed_config


def initialize(specs, parsed_config) -> gws.RootObject:
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

    r.post_initialize()

    return r


def activate(r: gws.RootObject):
    return gws.set_app_global(_ROOT_NAME, r)


def store(r: gws.RootObject, path=None):
    path = path or _STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.serialize_to_path(r, path)
    except Exception as e:
        raise error.LoadError('unable to store configuration') from e


def load(path=None) -> gws.RootObject:
    path = path or _STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        r = gws.unserialize_from_path(path)
        return gws.set_app_global(_ROOT_NAME, r)
    except Exception as e:
        raise error.LoadError('unable to load configuration') from e


def root() -> gws.RootObject:
    def _err():
        raise error.LoadError('no configuration root found')

    return gws.get_app_global(_ROOT_NAME, _err)


def real_config_path(config_path=None):
    p = config_path or os.getenv('GWS_CONFIG')
    if p:
        return p
    for p in DEFAULT_CONFIG_PATHS:
        if gws.is_file(p):
            return p

import os

import gws
import gws.spec.runtime
from . import parser

STORE_PATH = gws.CONFIG_DIR + '/config.pickle'
ROOT_NAME = 'gws_root_object'


def configure(
        manifest_path=None,
        config_path=None,
        config=None,
        before_init=None,
        fallback_config=None
) -> gws.RootObject:
    """Configure the server"""

    def _print(a):
        if isinstance(a, (list, tuple)):
            for item in a:
                _print(item)
        elif a is not None:
            for s in gws.lines(str(a)):
                gws.log.error(s)

    def _report(_args):
        _print('-' * 20)
        _print('CONFIGURATION ERROR:')
        _print('-' * 20)
        _print(_args)

    def _fallback(_exc, msg):
        if fallback_config and specs.manifest.withFallbackConfig:
            gws.log.warn(f'configuration error: using fallback config')
            return configure(config=fallback_config)
        raise gws.ConfigurationError(msg) from _exc

    if manifest_path:
        gws.log.info(f'using manifest {manifest_path!r}...')

    try:
        specs = gws.spec.runtime.load(manifest_path)
    except Exception as exc:
        # no fallback here
        _report(exc.args)
        raise gws.ConfigurationError('spec failed') from exc

    if not config:
        try:
            config = parser.parse_main(specs, config_path)
        except Exception as exc:
            _report(exc.args)
            return _fallback(exc, 'parse error')

    if before_init:
        try:
            before_init(config)
        except Exception as exc:
            _report(exc.args)
            return _fallback(exc, 'pre-init error')

    try:
        root_object = initialize(specs, config)
    except Exception as exc:
        _report(exc.args)
        return _fallback(exc, 'init error')

    if not root_object.configuration_errors:
        gws.log.info('configuration ok')
        return root_object

    for err in root_object.configuration_errors:
        _report(err)

    if specs.manifest.withStrictConfig:
        return _fallback(gws.ConfigurationError(), 'configuration failed')

    gws.log.warn(f'CONFIGURATION ERRORS: {len(root_object.configuration_errors)}')
    return root_object


def initialize(specs, parsed_config) -> gws.RootObject:
    r = gws.RootObject()
    r.specs = specs

    try:
        ts = gws.time_start('loading application')
        mod = gws.import_from_path(gws.APP_DIR + '/gws/base/application/__init__.py', 'gws.base.application')
        app = getattr(mod, 'Object')
        gws.time_end(ts)
    except Exception as e:
        raise gws.ConfigurationError(*e.args)

    try:
        ts = gws.time_start('configuring application')
        r.create_application(app, parsed_config)
        gws.time_end(ts)
    except Exception as e:
        raise gws.ConfigurationError(*e.args)

    r.post_initialize()

    return r


def activate(r: gws.RootObject):
    return gws.set_app_global(ROOT_NAME, r)


def deactivate():
    return gws.delete_app_global(ROOT_NAME)


def store(r: gws.RootObject, path=None):
    path = path or STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.serialize_to_path(r, path)
    except Exception as e:
        raise gws.ConfigurationError('unable to store configuration') from e


def load(path=None) -> gws.RootObject:
    path = path or STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        r = gws.unserialize_from_path(path)
        return gws.set_app_global(ROOT_NAME, r)
    except Exception as e:
        raise gws.ConfigurationError('unable to load configuration') from e


def root() -> gws.RootObject:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.get_app_global(ROOT_NAME, _err)

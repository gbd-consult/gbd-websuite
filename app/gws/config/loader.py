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
) -> gws.IRoot:
    """Configure the server"""

    def _print(a):
        if isinstance(a, (list, tuple)):
            for item in a:
                _print(item)
        elif a is not None:
            for s in gws.to_lines(str(a)):
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
        specs = gws.spec.runtime.create(manifest_path, read_cache=True, write_cache=True)
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

    if not root_object.configErrors:
        gws.log.info('configuration ok')
        return root_object

    for err in root_object.configErrors:
        _report(err)

    if specs.manifest.withStrictConfig:
        return _fallback(gws.ConfigurationError(), 'configuration failed')

    gws.log.warn(f'CONFIGURATION ERRORS: {len(root_object.configErrors)}')
    return root_object


def initialize(specs, parsed_config) -> gws.IRoot:
    r = gws.create_root_object(specs)

    try:
        ts = gws.time_start('configuring application')
        r.create_application(parsed_config)
        gws.time_end(ts)
    except Exception as exc:
        raise gws.ConfigurationError(*exc.args)

    r.post_initialize()

    return r


def activate(r: gws.IRoot):
    r.activate()
    return gws.set_app_global(ROOT_NAME, r)


def deactivate():
    return gws.delete_app_global(ROOT_NAME)


def store(r: gws.IRoot, path=None):
    path = path or STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.serialize_to_path(r, path)
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    path = path or STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        ts = gws.time_start('loading config')
        r = gws.unserialize_from_path(path)
        gws.time_end(ts)
        return activate(r)
    except Exception as exc:
        raise gws.ConfigurationError('unable to load configuration') from exc


def root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.get_app_global(ROOT_NAME, _err)

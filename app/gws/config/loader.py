import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx

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
            for s in str(a).split('\n'):
                gws.log.error(errpfx + s)

    def _report(_exc):
        _print(getattr(_exc, 'args', repr(_exc)))

    errors = []
    errpfx = 'CONFIGURATION ERROR: '
    root_object = None

    try:
        specs = gws.spec.runtime.create(manifest_path, read_cache=True, write_cache=True)
    except Exception as exc:
        _report(exc)
        raise gws.ConfigurationError('spec failed') from exc

    if not config:
        p = parser.ConfigParser(specs)
        config = p.parse_main(config_path)
        errors.extend(p.errors)

    if config:
        if before_init:
            before_init(config)
        root_object = initialize(specs, config)
        if root_object:
            errors.extend(root_object.configErrors)

    if errors:
        cnt = len(errors)
        for n, err in enumerate(errors, 1):
            gws.log.error(errpfx + f'{n} of {cnt}')
            _report(err)
            gws.log.error(errpfx + ('-' * 60))
        gws.log.error(errpfx + f'total {cnt} error(s)')

    if root_object:
        if not errors:
            gws.log.info(f'configuration ok, {root_object.object_count()} objects')
            return root_object
        if not specs.manifest.withStrictConfig:
            gws.log.warning(f'configuration complete with errors, {root_object.object_count()} objects')
            return root_object

    if specs.manifest.withFallbackConfig and fallback_config:
        gws.log.warning(f'using fallback config')
        return configure(config=fallback_config)

    raise gws.ConfigurationError('configuration failed')


def initialize(specs, parsed_config) -> gws.IRoot:
    r = gws.create_root_object(specs)

    try:
        gws.time_start('configuring application')
        r.create_application(parsed_config)
        gws.time_end()
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
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.serialize_to_path(r, path)
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    path = path or STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        sys_path = gws.lib.jsonx.from_path(f'{path}.syspath.json')
        for p in sys_path:
            if p not in sys.path:
                sys.path.insert(0, p)
                gws.log.debug(f'path {p!r} added to sys.path')

        gws.time_start('loading config')
        r = gws.unserialize_from_path(path)
        gws.time_end()
        return activate(r)
    except Exception as exc:
        raise gws.ConfigurationError('unable to load configuration') from exc


def root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.get_app_global(ROOT_NAME, _err)

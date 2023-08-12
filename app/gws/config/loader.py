import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx

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
    ro = None

    ts = gws.lib.osx.utime()
    ms = gws.lib.osx.process_rss_size()

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
        ro = initialize(specs, config)
        if ro:
            errors.extend(ro.configErrors)

    err_cnt = 0

    if errors:
        err_cnt = len(errors)
        for n, err in enumerate(errors, 1):
            gws.log.error(errpfx + f'{n} of {err_cnt}')
            _report(err)
            gws.log.error(errpfx + ('-' * 60))

    if ro:
        info = '{:d} objects, time: {:.2f} s., memory: {:.2f} MB'.format(
            ro.object_count(),
            gws.lib.osx.utime() - ts,
            gws.lib.osx.process_rss_size() - ms,
        )
        if not errors:
            gws.log.info(f'configuration ok, {info}')
            return ro
        if not specs.manifest.withStrictConfig:
            gws.log.warning(f'configuration complete with {err_cnt} error(s), {info}')
            return ro

    if specs.manifest.withFallbackConfig and fallback_config:
        gws.log.warning(f'using fallback config')
        return configure(config=fallback_config)

    raise gws.ConfigurationError('configuration failed')


def initialize(specs, parsed_config) -> gws.IRoot:
    ro = gws.create_root_object(specs)
    ro.create_application(parsed_config)
    ro.post_initialize()
    return ro


def activate(ro: gws.IRoot):
    ro.activate()
    return gws.set_app_global(ROOT_NAME, ro)


def deactivate():
    return gws.delete_app_global(ROOT_NAME)


def store(ro: gws.IRoot, path=None):
    path = path or STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.serialize_to_path(ro, path)
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    path = path or STORE_PATH
    gws.log.debug(f'loading config from {path!r}')
    try:
        return _load(path)
    except Exception as exc:
        raise gws.ConfigurationError('unable to load configuration') from exc


def _load(path) -> gws.Root:
    sys_path = gws.lib.jsonx.from_path(f'{path}.syspath.json')
    for p in sys_path:
        if p not in sys.path:
            sys.path.insert(0, p)
            gws.log.debug(f'path {p!r} added to sys.path')

    ts = gws.lib.osx.utime()
    ms = gws.lib.osx.process_rss_size()

    ro = gws.unserialize_from_path(path)

    activate(ro)

    info = 'configuration ok, {:d} objects, time: {:.2f} s., memory: {:.2f} MB'.format(
        ro.object_count(),
        gws.lib.osx.utime() - ts,
        gws.lib.osx.process_rss_size() - ms,
    )

    gws.log.info(info)

    return ro


def root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.get_app_global(ROOT_NAME, _err)

import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx

from . import parser

STORE_PATH = gws.c.CONFIG_DIR + '/config.pickle'
ROOT_NAME = 'gws_root_object'


def configure(
        manifest_path=None,
        config_path=None,
        config=None,
        before_init=None,
        fallback_config=None,
        with_spec_cache=True,
) -> gws.Root:
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
    root_obj = None

    ts = gws.lib.osx.utime()
    ms = gws.lib.osx.process_rss_size()

    manifest_path = real_manifest_path(manifest_path)
    if manifest_path:
        gws.log.info(f'using manifest {manifest_path!r}...')

    try:
        specs = gws.spec.runtime.create(
            manifest_path=manifest_path,
            read_cache=with_spec_cache,
            write_cache=with_spec_cache
        )
    except Exception as exc:
        _report(exc)
        raise gws.ConfigurationError('spec failed') from exc

    if not config:
        config_path = real_config_path(config_path)
        if config_path:
            gws.log.info(f'using config {config_path!r}...')
            p = parser.ConfigParser(specs)
            config = p.parse_main(config_path)
            errors.extend(p.errors)
        else:
            errors.append(gws.ConfigurationError('no configuration file found'))

    if config:
        if before_init:
            before_init(config)
        root_obj = initialize(specs, config)
        if root_obj:
            errors.extend(root_obj.configErrors)

    err_cnt = 0

    if errors:
        err_cnt = len(errors)
        for n, err in enumerate(errors, 1):
            gws.log.error(errpfx + f'{n} of {err_cnt}')
            _report(err)
            gws.log.error(errpfx + ('-' * 60))

    if root_obj:
        info = '{:d} objects, time: {:.2f} s., memory: {:.2f} MB'.format(
            root_obj.object_count(),
            gws.lib.osx.utime() - ts,
            gws.lib.osx.process_rss_size() - ms,
        )
        if not errors:
            gws.log.info(f'configuration ok, {info}')
            return root_obj
        if not specs.manifest.withStrictConfig:
            gws.log.warning(f'configuration complete with {err_cnt} error(s), {info}')
            return root_obj

    if specs.manifest.withFallbackConfig and fallback_config:
        gws.log.warning(f'using fallback config')
        return configure(config=fallback_config)

    raise gws.ConfigurationError('configuration failed')


def initialize(specs, parsed_config) -> gws.Root:
    root_obj = gws.u.create_root(specs)
    root_obj.create_application(parsed_config)
    root_obj.post_initialize()
    return root_obj


def activate(root_obj: gws.Root):
    root_obj.activate()
    return gws.u.set_app_global(ROOT_NAME, root_obj)


def deactivate():
    return gws.u.delete_app_global(ROOT_NAME)


def store(root_obj: gws.Root, path=None) -> str:
    path = path or STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.u.serialize_to_path(root_obj, path)
        return path
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

    root_obj = gws.u.unserialize_from_path(path)

    activate(root_obj)

    gws.log.info('configuration ok, {:d} objects, time: {:.2f} s., memory: {:.2f} MB'.format(
        root_obj.object_count(),
        gws.lib.osx.utime() - ts,
        gws.lib.osx.process_rss_size() - ms,
    ))

    return root_obj


def root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.u.get_app_global(ROOT_NAME, _err)


_DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]


def real_config_path(config_path):
    p = config_path or gws.env.GWS_CONFIG
    if p:
        return p
    for p in _DEFAULT_CONFIG_PATHS:
        if gws.u.is_file(p):
            return p


_DEFAULT_MANIFEST_PATHS = [
    '/data/MANIFEST.json',
]


def real_manifest_path(manifest_path):
    p = manifest_path or gws.env.GWS_MANIFEST
    if p:
        return p
    for p in _DEFAULT_MANIFEST_PATHS:
        if gws.u.is_file(p):
            return p

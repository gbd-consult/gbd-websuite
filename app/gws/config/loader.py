from typing import Optional
import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx

from . import parser

_ERROR_PREFIX = 'CONFIGURATION ERROR'

_ROOT_NAME = 'gws_root_object'

_DEFAULT_STORE_PATH = gws.c.CONFIG_DIR + '/config.pickle'

_DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]

_DEFAULT_MANIFEST_PATHS = [
    '/data/MANIFEST.json',
]


class Configurer:
    def __init__(
            self,
            manifest_path=None,
            config_path=None,
            config=None,
            before_init=None,
            fallback_config=None,
            with_spec_cache=True,
    ):
        self.manifest_path = real_manifest_path(manifest_path)
        if self.manifest_path:
            gws.log.info(f'using manifest {self.manifest_path!r}...')

        self.config_path = real_config_path(config_path)
        self.config = config
        self.before_init = before_init
        self.fallback_config = fallback_config
        self.with_spec_cache = with_spec_cache

        self.errors = []
        self.start_time = gws.u.stime()
        self.start_mem = gws.lib.osx.process_rss_size()

        self.specs = None
        self.root_obj = None

    def configure(self):
        self.specs = self._init_specs()
        if not self.specs:
            self._print_report(False)
            return

        self.config = self._init_config()
        if self.config:
            self.root_obj = self._init_root()

        if not self.root_obj and self.specs.manifest.withFallbackConfig and self.fallback_config:
            gws.log.warning(f'using fallback config')
            self.config = self.fallback_config
            self.root_obj = self._init_root()

        if not self.root_obj:
            self._print_report(False)
            return

        if self.errors and self.specs.manifest.withStrictConfig:
            self._print_report(False)
            return

        self._print_report(True)
        return self.root_obj

    def parse(self):
        self.specs = self._init_specs()
        if not self.specs:
            self._print_report(False)
            return

        self.config = self._init_config()
        if not self.config:
            self._print_report(False)
            return

        self._print_report(True)
        return self.config

    ##

    def _init_specs(self):
        try:
            return gws.spec.runtime.create(
                manifest_path=self.manifest_path,
                read_cache=self.with_spec_cache,
                write_cache=self.with_spec_cache
            )
        except Exception as exc:
            gws.log.exception()
            self._error(exc)

    def _init_config(self):
        if self.config:
            return self.config

        if not self.config_path:
            self._error(gws.ConfigurationError('no configuration file found'))
            return

        gws.log.info(f'using config {self.config_path!r}...')
        p = parser.ConfigParser(self.specs)
        config = p.parse_main(self.config_path)
        for err in p.errors:
            self._error(err)

        return config

    def _init_root(self):
        if self.before_init:
            self.before_init(self.config)
        root = initialize(self.specs, self.config)
        if root:
            for err in root.configErrors:
                self._error(err)
        return root

    def _error(self, exc):
        self.errors.append(getattr(exc, 'args', repr(exc)))

    def _print_report(self, ok):
        err_cnt = len(self.errors)

        if err_cnt > 0:
            for n, err in enumerate(self.errors, 1):
                gws.log.error(f'{_ERROR_PREFIX}: {n} of {err_cnt}')
                self._log(err)
                gws.log.error(f'{_ERROR_PREFIX}: {"-" * 60}')

        info = '{:d} objects, time: {:d} s., memory: {:.2f} MB'.format(
            self.root_obj.object_count() if self.root_obj else 0,
            gws.u.stime() - self.start_time,
            gws.lib.osx.process_rss_size() - self.start_mem,
        )

        if not ok:
            gws.log.error(f'configuration FAILED, {err_cnt} errors, {info}')
            return

        if err_cnt > 0:
            gws.log.warning(f'configuration complete,  {err_cnt} error(s), {info}')
            return

        gws.log.info(f'configuration ok, {info}')

    def _log(self, a):
        if isinstance(a, (list, tuple)):
            for item in a:
                self._log(item)
        elif a is not None:
            for s in str(a).split('\n'):
                gws.log.error(f'{_ERROR_PREFIX}: {s}')


def configure(
        manifest_path=None,
        config_path=None,
        config=None,
        before_init=None,
        fallback_config=None,
        with_spec_cache=True,
) -> Optional[gws.Root]:
    """Configure the server, return the Root object."""

    cc = Configurer(
        manifest_path,
        config_path,
        config,
        before_init,
        fallback_config,
        with_spec_cache
    )
    return cc.configure()


def parse(manifest_path=None, config_path=None) -> Optional[gws.Config]:
    """Parse input configuration and return a config object. """

    cc = Configurer(manifest_path, config_path)
    return cc.parse()


def initialize(specs: gws.spec.runtime.Object, config: gws.Config) -> gws.Root:
    root_obj = gws.create_root(specs)
    root_obj.create_application(config)
    root_obj.post_initialize()
    return root_obj


def activate(root_obj: gws.Root):
    root_obj.activate()
    return gws.u.set_app_global(_ROOT_NAME, root_obj)


def deactivate():
    return gws.u.delete_app_global(_ROOT_NAME)


def store(root_obj: gws.Root, path=None) -> str:
    path = path or _DEFAULT_STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.u.serialize_to_path(root_obj, path)
        return path
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    path = path or _DEFAULT_STORE_PATH
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

    ts = gws.u.stime()
    ms = gws.lib.osx.process_rss_size()

    root_obj = gws.u.unserialize_from_path(path)

    activate(root_obj)

    gws.log.info('configuration ok, {:d} objects, time: {:d} s., memory: {:.2f} MB'.format(
        root_obj.object_count(),
        gws.u.stime() - ts,
        gws.lib.osx.process_rss_size() - ms,
    ))

    return root_obj


def get_root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.u.get_app_global(_ROOT_NAME, _err)


def real_config_path(config_path):
    p = config_path or gws.env.GWS_CONFIG
    if p:
        return p
    for p in _DEFAULT_CONFIG_PATHS:
        if gws.u.is_file(p):
            return p


def real_manifest_path(manifest_path):
    p = manifest_path or gws.env.GWS_MANIFEST
    if p:
        return p
    for p in _DEFAULT_MANIFEST_PATHS:
        if gws.u.is_file(p):
            return p

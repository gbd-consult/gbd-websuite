from typing import Optional
import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.importer

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


class Object:
    manifestPath: str
    configPath: str
    fallbackConfig: Optional[gws.Config]
    withSpecCache: bool
    isStarting: bool

    manifest: Optional[gws.ApplicationManifest]
    config: Optional[gws.Config]
    specs: Optional[gws.spec.runtime.Object]
    root: Optional[gws.Root]

    errors: list

    def __init__(
        self,
        manifest_path: str = '',
        config_path: str = '',
        config=None,
        fallback_config=None,
        with_spec_cache=True,
        is_starting=False,
    ):
        self.manifestPath = real_manifest_path(manifest_path)
        if self.manifestPath:
            gws.log.info(f'using manifest {self.manifestPath!r}...')
        self.configPath = real_config_path(config_path)
        self.fallbackConfig = fallback_config
        self.withSpecCache = with_spec_cache
        self.isStarting = is_starting

        self.config = config
        self.specs = None
        self.manifest = None
        self.root = None

        self.allConfigPaths = set([self.configPath])

        self.errors = []
        self.tm1 = _time_and_memory()

    def configure(self):
        self.specs = self._create_specs()
        if not self.specs:
            self._print_report(False)
            return

        self.manifest = self.specs.manifest

        self.config = self._create_config()
        if self.config:
            self._init_root_with_hooks()

        if not self.root and self.specs.manifest.withFallbackConfig and self.fallbackConfig:
            gws.log.warning(f'using fallback config')
            self.config = self.fallbackConfig
            self.root = self._create_root()

        if not self.root:
            self._print_report(False)
            return

        if self.errors and self.specs.manifest.withStrictConfig:
            self._print_report(False)
            return

        self._print_report(True)

        setattr(self.root.app.config, 'configPaths', list(self.allConfigPaths))
        return self.root

    def parse(self):
        self.specs = self._create_specs()
        if not self.specs:
            self._print_report(False)
            return

        self.config = self._create_config()
        if not self.config:
            self._print_report(False)
            return

        self._print_report(True)
        return self.config

    ##

    def _create_specs(self):
        if self.specs:
            return self.specs
        if not self.manifestPath:
            self._error(gws.ConfigurationError('no manifest file found'))
        try:
            return gws.spec.runtime.create(
                manifest_path=self.manifestPath,
                read_cache=self.withSpecCache,
                write_cache=self.withSpecCache
            )
        except Exception as exc:
            gws.log.exception()
            self._error(exc)

    def _create_config(self):
        if self.config:
            return self.config

        if not self.configPath:
            self._error(gws.ConfigurationError('no configuration file found'))
            return

        gws.log.info(f'using config {self.configPath!r}...')
        pr = parser.parse_app_config_path(self.configPath, gws.u.require(self.specs))
        for err in pr.errors:
            self._error(err)
        self.allConfigPaths.update(pr.paths)

        return pr.config

    def _create_root(self):
        root = initialize(gws.u.require(self.specs), gws.u.require(self.config))
        if root:
            for err in root.configErrors:
                self._error(err)
        return root

    ##

    def _init_root_with_hooks(self):
        try:
            self._pre_configure()
        except Exception as exc:
            gws.log.exception()
            self._error(f'preConfigure failed: {exc!r}')
            return

        self.root = self._create_root()

        try:
            self._post_configure()
        except Exception as exc:
            gws.log.exception()
            self._error(f'postConfigure failed: {exc!r}')
            return

    def _pre_configure(self):
        # 'server.autoRun' bash script - only when starting
        if self.isStarting:
            autorun = gws.u.get(self.config, 'server.autoRun')
            if autorun:
                gws.log.info(f'running autorun: {autorun!r}')
                gws.lib.osx.run(autorun, echo=True)

        self._run_hook('preConfigure')
        parser.save_debug(self.config, self.configPath, '.preconf.json')

    def _post_configure(self):
        self._run_hook('postConfigure')

    def _run_hook(self, name):
        path = gws.u.get(self.config, f'server.{name}')
        if not path:
            return

        gws.log.info(f'running hook {name}: {path!r}')
        self.allConfigPaths.add(path)

        if path.endswith('.py'):
            fn = gws.lib.importer.load_file(path).get('main')
            if not fn:
                raise gws.ConfigurationError(f'invalid {name} hook: {path!r}')
            fn(self)
            return

        if path.endswith('.sh'):
            gws.lib.osx.run(f'bash {path}', echo=True)
            return

        raise gws.ConfigurationError(f'invalid {name} hook: {path!r}')

    ##

    def _error(self, exc):
        self.errors.append(getattr(exc, 'args', repr(exc)))

    def _print_report(self, ok):
        err_cnt = len(self.errors)

        if err_cnt > 0:
            for n, err in enumerate(self.errors, 1):
                gws.log.error(f'{_ERROR_PREFIX}: {n} of {err_cnt}')
                self._log(err)
                gws.log.error(f'{_ERROR_PREFIX}: {"-" * 60}')

        info = _info_string(self.root, self.tm1)

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
    manifest_path='',
    config_path='',
    config=None,
    fallback_config=None,
    with_spec_cache=True,
    is_starting=False,
) -> Optional[gws.Root]:
    """Configure the server, return the Root object."""

    cc = Object(
        manifest_path,
        config_path,
        config,
        fallback_config,
        with_spec_cache,
        is_starting,
    )
    return cc.configure()


def parse(manifest_path='', config_path='') -> Optional[gws.Config]:
    """Parse input configuration and return a config object. """

    cc = Object(manifest_path, config_path)
    return cc.parse()


def initialize(specs: gws.SpecRuntime, config: gws.Config) -> gws.Root:
    root = gws.create_root(specs)
    root.create_application(config)
    root.post_initialize()
    return root


def activate(root: gws.Root):
    root.activate()
    return gws.u.set_app_global(_ROOT_NAME, root)


def deactivate():
    return gws.u.delete_app_global(_ROOT_NAME)


def store(root: gws.Root, path=None) -> str:
    path = path or _DEFAULT_STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.u.serialize_to_path(root, path)
        return path
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    ui = gws.lib.osx.user_info()
    path = path or _DEFAULT_STORE_PATH
    gws.log.info(f"loading config from {path!r}, user {ui['pw_name']} ({ui['pw_uid']}:{ui['pw_gid']})")
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

    tm1 = _time_and_memory()

    root = gws.u.unserialize_from_path(path)

    activate(root)

    info = _info_string(root, tm1)
    gws.log.info(f'configuration loaded, {info}')

    return root


def get_root() -> gws.Root:
    def _err():
        raise gws.Error('no configuration root found')

    return gws.u.get_app_global(_ROOT_NAME, _err)


def real_config_path(config_path: str) -> str:
    p = config_path or gws.env.GWS_CONFIG
    if p:
        return p
    for p in _DEFAULT_CONFIG_PATHS:
        if gws.u.is_file(p):
            return p
    return ''


def real_manifest_path(manifest_path: str) -> str:
    p = manifest_path or gws.env.GWS_MANIFEST
    if p:
        return p
    for p in _DEFAULT_MANIFEST_PATHS:
        if gws.u.is_file(p):
            return p
    return ''


def _time_and_memory():
    return gws.u.stime(), gws.lib.osx.process_rss_size()


def _info_string(root, tm1):
    tm2 = _time_and_memory()
    return '{:d} objects, time: {:d} s., memory: {:.2f} MB'.format(
        root.object_count() if root else 0,
        tm2[0] - tm1[0],
        tm2[1] - tm1[1],
    )

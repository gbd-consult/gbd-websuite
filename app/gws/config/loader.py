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
    ctx: gws.ConfigContext
    manifestPath: str
    configPath: str
    fallbackConfig: Optional[gws.Config]
    withSpecCache: bool

    def __init__(
        self,
        manifest_path: str = '',
        config_path: str = '',
        raw_config=None,
        fallback_config=None,
        with_spec_cache=True,
        hooks: list = None,
    ):
        self.tm1 = _time_and_memory()

        self.ctx = gws.ConfigContext(
            errors=[],
        )

        self.manifestPath = real_manifest_path(manifest_path)
        if self.manifestPath:
            gws.log.info(f'using manifest {self.manifestPath!r}...')

        self.configPath = real_config_path(config_path)
        self.rawConfig = raw_config
        self.fallbackConfig = fallback_config
        self.withSpecCache = with_spec_cache
        self.hooks = hooks or []

        self.config = None
        self.root = None

    def configure(self) -> Optional[gws.Root]:
        if not self._init_specs():
            self._print_report(ok=False)
            return

        self._run_hook('preConfigure')
        if not self.config:
            self.config = self._create_config()
        self._run_hook('postConfigure')

        if self.config:
            self._run_hook('preInitialize')
            if not self.root:
                self.root = self._create_root(self.config)
            self._run_hook('postInitialize')

        if not self.root and self.ctx.specs.manifest.withFallbackConfig and self.fallbackConfig:
            gws.log.warning(f'using fallback config')
            self.root = self._create_root(self.fallbackConfig)

        if not self.root:
            self._print_report(ok=False)
            return

        if self.ctx.errors and self.ctx.specs.manifest.withStrictConfig:
            self._print_report(ok=False)
            return

        self._print_report(ok=True)

        self.root.configPaths = list(self.ctx.paths)
        return self.root

    def parse(self) -> Optional[gws.Config]:
        if not self._init_specs():
            self._print_report(ok=False)
            return

        cfg = self._create_config()
        if not cfg:
            self._print_report(ok=False)
            return

        self._print_report(ok=True)
        return cfg

    ##

    def _init_specs(self):
        if not self.manifestPath:
            self._error(gws.ConfigurationError('no manifest file found'))
        try:
            self.ctx.specs = gws.spec.runtime.create(
                manifest_path=self.manifestPath,
                # read_cache=self.withSpecCache,
                # write_cache=self.withSpecCache,
            )
            return True
        except Exception as exc:
            gws.log.exception()
            self._error(exc)
            return False

    def _create_config(self):
        if self.rawConfig:
            return parser.parse_app_dict(self.rawConfig, '', self.ctx)
        if not self.configPath:
            self._error(gws.ConfigurationError('no configuration file found'))
            return
        gws.log.info(f'using config {self.configPath!r}...')
        return parser.parse_app_from_path(self.configPath, self.ctx)

    def _create_root(self, cfg):
        root = initialize(self.ctx.specs, cfg)
        if root:
            for err in root.configErrors:
                self._error(err)
        return root

    def _run_hook(self, event):
        for evt, fn in self.hooks:
            if event != evt:
                continue
            try:
                fn(self)
            except Exception as exc:
                gws.log.exception()
                self._error(exc)

    def _error(self, exc):
        self.ctx.errors.append(getattr(exc, 'args', repr(exc)))

    def _print_report(self, ok):
        err_cnt = len(self.ctx.errors)

        if err_cnt > 0:
            for n, err in enumerate(self.ctx.errors, 1):
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
            return

        if hasattr(a, 'args'):
            lines = [str(x) for x in a.args]
        else:
            lines = str(a).split('\n')

        for s in lines:
            gws.log.error(f'{_ERROR_PREFIX}: {s}')


def configure(
    manifest_path='',
    config_path='',
    raw_config=None,
    fallback_config=None,
    with_spec_cache=True,
    hooks: list = None,
) -> Optional[gws.Root]:
    """Configure the server, return the Root object."""

    ldr = Object(
        manifest_path,
        config_path,
        raw_config,
        fallback_config,
        with_spec_cache,
        hooks,
    )
    return ldr.configure()


def parse(manifest_path='', config_path='') -> Optional[gws.Config]:
    """Parse input configuration and return a config object."""

    ldr = Object(
        manifest_path,
        config_path,
    )
    return ldr.parse()


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
    gws.log.info(f'loading config from {path!r}, user {ui["pw_name"]} ({ui["pw_uid"]}:{ui["pw_gid"]})')
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

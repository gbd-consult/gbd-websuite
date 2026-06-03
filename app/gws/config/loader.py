from typing import Optional
import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.dynimport

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
        manifest_path='',
        config_path='',
        specs=None,
        raw_config=None,
        fallback_config=None,
        with_spec_cache=False,
        hooks=None,
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
        self.specs = specs

        self.config = None
        self.root = None

    def configure(self) -> gws.ConfigResult:
        if not self._init_specs():
            return self._result()

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
            return self._result()

        if self.ctx.errors and self.ctx.specs.manifest.withStrictConfig:
            self.root = None
            return self._result()

        self.root.configPaths = list(self.ctx.paths)
        return self._result()

    def parse(self) -> gws.ConfigResult:
        if not self._init_specs():
            return self._result()

        self.config = self._create_config()
        if not self.config:
            return self._result()

        return self._result()

    ##

    def _init_specs(self):
        if self.specs:
            self.ctx.specs = self.specs
            return True

        try:
            self.ctx.specs = gws.spec.runtime.create(
                manifest_path=self.manifestPath,
                read_cache=self.withSpecCache,
                write_cache=self.withSpecCache,
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
            for ce in root.configErrors:
                self.ctx.errors.append(gws.ConfigErrorInfo(ce))
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
        cei = gws.ConfigErrorInfo(message=str(exc))
        if exc.__cause__:
            cei.cause = repr(exc.__cause__)
        self.ctx.errors.append(cei)

    def _result(self):
        return gws.ConfigResult(
            errors=self.ctx.errors,
            root=self.root,
            config=self.config,
            info=_info_string(self.root, self.tm1),
        )


def configure(
    manifest_path='',
    config_path='',
    specs: Optional[gws.SpecRuntime] = None,
    raw_config: dict | gws.Data = None,
    fallback_config: dict | gws.Data = None,
    with_spec_cache=False,
    hooks: list = None,
) -> gws.ConfigResult:
    """Configure the server."""

    ldr = Object(
        manifest_path,
        config_path,
        specs,
        raw_config,
        fallback_config,
        with_spec_cache,
        hooks,
    )
    return ldr.configure()


def parse(
    manifest_path='',
    config_path='',
    specs: Optional[gws.SpecRuntime] = None,
) -> gws.ConfigResult:
    """Parse input configuration."""

    ldr = Object(
        manifest_path,
        config_path,
        specs,
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
        for s in p.split(','):
            s = s.strip()
            if gws.u.is_file(s):
                return s
        return ''
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


def log_report(cr: gws.ConfigResult):
    err_cnt = len(cr.errors) if cr.errors else 0
    ln = '*' * 80

    if err_cnt == 0:
        gws.log.info(ln)
        gws.log.info(f'configured: {cr.info}')
        gws.log.info(ln)
        return

    gws.log.error(ln)
    gws.log.error(f'configured wth errors: errors: {err_cnt}, {cr.info}')
    gws.log.error(ln)

    for n, cei in enumerate(cr.errors, 1):
        gws.log.error(f'{_ERROR_PREFIX}: {n} of {err_cnt}')
        _log_error(cei)
        gws.log.error(f'{_ERROR_PREFIX}: ')

    gws.log.error(ln)


def _log_error(cei: gws.ConfigErrorInfo):
    ls = []
    ls.append(cei.message)
    tab = ' ' * 4

    if cei.path:
        ls.append(f'PATH:  {cei.path}')
    if cei.line:
        ls.append(f'LINE:  {cei.line}')
    if cei.value:
        ls.append(f'VALUE: {cei.value}')
    if cei.cause:
        ls.append(f'CAUSE: {cei.cause}')
    if cei.stack:
        for loc in cei.stack:
            p = [
                loc.objectType,
                repr(loc.objectName) if loc.objectName else None,
                f'uid={loc.objectUid}' if loc.objectUid else None,
            ]
            p = '<' + ' '.join(gws.u.compact(p)) + '>'
            if loc.propName:
                p = f'{loc.propName!r} {p}'
            ls.append(f'{tab}in {p}')
    if cei.contextLines:
        ls.extend(cei.contextLines)

    for s in ls:
        gws.log.error(f'{_ERROR_PREFIX}: {s}')


def _time_and_memory():
    return gws.u.stime(), gws.lib.osx.process_rss_size()


def _info_string(root, tm1):
    tm2 = _time_and_memory()
    return 'objects: {:d}, time: {:d}s., memory: {:.2f} MB'.format(
        root.object_count() if root else 0,
        tm2[0] - tm1[0],
        tm2[1] - tm1[1],
    )

"""Configuration loader: configure, store and load the root object."""

from typing import Optional
import sys

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.dynimport

from . import parser

_ERROR_PREFIX = 'CONFIGURATION ERROR'
_WARNING_PREFIX = 'CONFIGURATION WARNING'

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
    """Configuration loader.

    Holds the state of one configuration run: the specs, the parsed config,
    the root object and the collected errors and warnings.
    """

    ctx: gws.ConfigContext
    """Context with specs, errors, warnings and read paths."""
    manifestPath: str
    """Path to the application manifest, or an empty string."""
    configPath: str
    """Path to the configuration file, or an empty string."""
    fallbackConfig: Optional[gws.Config]
    """Configuration used when the main configuration fails, if the manifest allows it."""
    withSpecCache: bool
    """Read and write the specs cache."""

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
        """Create a loader.

        Args:
            manifest_path: Path to the application manifest. Defaults to ``GWS_MANIFEST`` or a default path.
            config_path: Path to the configuration file. Defaults to ``GWS_CONFIG`` or a default path.
            specs: Specs to use. If not given, specs are created from the manifest.
            raw_config: Configuration dict, used instead of the configuration file.
            fallback_config: Configuration used when the main configuration fails.
            with_spec_cache: Read and write the specs cache.
            hooks: List of ``(event, fn)`` pairs, see ``configure``.
        """
        self.tm1 = _time_and_memory()

        self.ctx = gws.ConfigContext(
            errors=[],
            warnings=[],
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
        """Parse the configuration and create the root object.

        Hooks are called with the loader object at the events ``preConfigure``,
        ``postConfigure`` (around parsing), ``preInitialize`` and ``postInitialize``
        (around creating the root). Exceptions in hooks are recorded as errors.

        If no root can be created, the fallback config is used, provided the
        manifest sets ``withFallbackConfig``. If there are errors and the manifest
        sets ``withStrictConfig``, the root is discarded.

        Returns:
            The configuration result.
        """
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
        """Parse and validate the configuration without creating objects.

        Returns:
            The configuration result, without the root object.
        """
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
            for cw in root.configWarnings:
                self.ctx.warnings.append(gws.ConfigErrorInfo(cw))
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
            warnings=self.ctx.warnings,
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
    """Parse the configuration and create the root object.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.
        specs: Specs to use. If not given, specs are created from the manifest.
        raw_config: Configuration dict, used instead of the configuration file.
        fallback_config: Configuration used when the main configuration fails.
        with_spec_cache: Read and write the specs cache.
        hooks: List of ``(event, fn)`` pairs, see ``Object.configure``.

    Returns:
        The configuration result. Its ``root`` is ``None`` if configuration failed.
    """

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
    """Parse and validate the configuration without creating objects.

    Args:
        manifest_path: Path to the application manifest.
        config_path: Path to the configuration file.
        specs: Specs to use. If not given, specs are created from the manifest.

    Returns:
        The configuration result with the parsed config, errors and warnings.
    """

    ldr = Object(
        manifest_path,
        config_path,
        specs,
    )
    return ldr.parse()


def initialize(specs: gws.SpecRuntime, config: gws.Config) -> gws.Root:
    """Create the root object and the application from a parsed configuration.

    Args:
        specs: Specs runtime.
        config: Parsed application configuration.

    Returns:
        The initialized, not yet activated root object.
    """
    root = gws.create_root(specs)
    root.create_application(config)
    root.post_initialize()
    return root


def activate(root: gws.Root):
    """Activate the root object and make it the current root.

    Args:
        root: Root object.

    Returns:
        The root object.
    """
    root.activate()
    return gws.u.set_app_global(_ROOT_NAME, root)


def deactivate():
    """Remove the current root."""
    return gws.u.delete_app_global(_ROOT_NAME)


def store(root: gws.Root, path=None) -> str:
    """Serialize the root object to a file.

    The current ``sys.path`` is saved next to it in ``<path>.syspath.json``,
    so that ``load`` can restore it.

    Args:
        root: Root object.
        path: File path. Defaults to ``config.pickle`` in the config directory.

    Returns:
        The file path.

    Raises:
        ``gws.ConfigurationError``: If the root cannot be stored.
    """
    path = path or _DEFAULT_STORE_PATH
    gws.log.debug(f'writing config to {path!r}')
    try:
        gws.lib.jsonx.to_path(f'{path}.syspath.json', sys.path)
        gws.u.serialize_to_path(root, path)
        return path
    except Exception as exc:
        raise gws.ConfigurationError('unable to store configuration') from exc


def load(path=None) -> gws.Root:
    """Load a stored root object, activate it and make it the current root.

    Args:
        path: File path. Defaults to ``config.pickle`` in the config directory.

    Returns:
        The root object.

    Raises:
        ``gws.ConfigurationError``: If the root cannot be loaded or activated.
    """
    ui = gws.lib.osx.user_info()
    path = path or _DEFAULT_STORE_PATH
    gws.log.info(f'loading config from {path!r}, user {ui["pw_name"]} ({ui["pw_uid"]}:{ui["pw_gid"]})')
    try:
        return _load(path)
    except Exception as exc:
        raise gws.ConfigurationError('unable to load configuration') from exc


def _load(path) -> gws.Root:
    """Restore ``sys.path``, unserialize and activate the root."""
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
    """Return the current root object.

    Returns:
        The root object set by ``activate`` or ``load``.

    Raises:
        ``gws.Error``: If there is no current root.
    """
    def _err():
        raise gws.Error('no configuration root found')

    return gws.u.get_app_global(_ROOT_NAME, _err)


def real_config_path(config_path: str) -> str:
    """Find the configuration file.

    Args:
        config_path: Comma-separated list of paths. Defaults to the ``GWS_CONFIG`` environment variable.
            If neither is given, the default paths are checked.

    Returns:
        The first path that is an existing file, or an empty string.
    """
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
    """Find the application manifest.

    Args:
        manifest_path: Manifest path. Defaults to the ``GWS_MANIFEST`` environment variable.
            If neither is given, the default path ``/data/MANIFEST.json`` is used if it exists.

    Returns:
        The manifest path, or an empty string.
    """
    p = manifest_path or gws.env.GWS_MANIFEST
    if p:
        return p
    for p in _DEFAULT_MANIFEST_PATHS:
        if gws.u.is_file(p):
            return p
    return ''


def log_report(cr: gws.ConfigResult):
    """Log a summary of the configuration result and all errors and warnings.

    Args:
        cr: Configuration result.
    """
    err_cnt = len(cr.errors) if cr.errors else 0
    warn_cnt = len(cr.warnings) if cr.warnings else 0
    ln = '*' * 80

    if err_cnt == 0 and warn_cnt == 0:
        gws.log.info(ln)
        gws.log.info(f'configured: {cr.info}')
        gws.log.info(ln)
        return

    if err_cnt == 0:
        gws.log.warning(ln)
        gws.log.warning(f'configured with warnings: {warn_cnt}, {cr.info}')
        gws.log.warning(ln)
    else:
        gws.log.error(ln)
        gws.log.error(f'configured with errors: {err_cnt}, warnings: {warn_cnt}, {cr.info}')
        gws.log.error(ln)

    # cr.errors.sort(key=lambda ce: ce.message)

    for n, cei in enumerate(cr.errors, 1):
        gws.log.error(f'{_ERROR_PREFIX}: {n} of {err_cnt}')
        _log_info(cei, gws.log.error, _ERROR_PREFIX)
        gws.log.error(f'{_ERROR_PREFIX}: ')

    for n, cei in enumerate(cr.warnings, 1):
        gws.log.warning(f'{_WARNING_PREFIX}: {n} of {warn_cnt}')
        _log_info(cei, gws.log.warning, _WARNING_PREFIX)
        gws.log.warning(f'{_WARNING_PREFIX}: ')

    if err_cnt == 0:
        gws.log.warning(ln)
    else:
        gws.log.error(ln)


def _log_info(cei: gws.ConfigErrorInfo, log_fn, prefix):
    """Log the details of an error or warning."""
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
        log_fn(f'{prefix}: {s}')


def _time_and_memory():
    return gws.u.stime(), gws.lib.osx.process_rss_size()


def _info_string(root, tm1):
    tm2 = _time_and_memory()
    return 'objects: {:d}, time: {:d}s., memory: {:.2f} MB'.format(
        root.object_count() if root else 0,
        tm2[0] - tm1[0],
        tm2[1] - tm1[1],
    )

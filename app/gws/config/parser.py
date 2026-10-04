"""Configuration parser: read configuration files and validate them against the specs."""

from typing import Optional, cast

import os
import yaml

import gws
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.datetimex
import gws.lib.dynimport
import gws.lib.vendor.jump
import gws.lib.vendor.slon
import gws.spec.runtime

CONFIG_PATH_PATTERN = r'\.(py|json|yaml|yml|cx)$'


def parse_from_path(path: str, as_type: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Read a configuration file and validate it as a given type.

    Args:
        path: Path to the configuration file.
        as_type: Type of the configuration, e.g. ``gws.base.application.core.Config``.
        ctx: Configuration context. Errors are added to it.

    Returns:
        The parsed configuration, or ``None`` if there were errors.
    """

    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return pp.parse_dict(d, path, as_type) if d else None


def parse_dict(dct: dict | gws.Data, path: str, as_type: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Validate a configuration dict as a given type.

    Args:
        dct: Configuration dict or ``gws.Data`` object.
        path: Path to the configuration file, for error reporting.
        as_type: Type of the configuration, e.g. ``gws.ext.config.layer``.
        ctx: Configuration context. Errors are added to it.

    Returns:
        The parsed configuration, or ``None`` if there were errors.
    """

    pp = _Parser(ctx)
    d = pp.ensure_dict(dct, path)
    return pp.parse_dict(d, path, as_type) if d else None


def parse_app_from_path(path: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Read and validate the application configuration file.

    Sets the server time zone, then parses the application config and all
    projects (inline ``projects``, ``projectPaths`` and files in ``projectDirs``).

    Args:
        path: Path to the application configuration file.
        ctx: Configuration context. Errors are added to it.

    Returns:
        The parsed application configuration, or ``None`` if it could not be parsed.
    """

    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return _parse_app_dict(d, path, pp) if d else None


def parse_app_dict(dct: dict | gws.Data, path: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Validate an application configuration dict.

    Works like ``parse_app_from_path``.

    Args:
        dct: Application configuration dict or ``gws.Data`` object.
        path: Path to the configuration file, for error reporting.
        ctx: Configuration context. Errors are added to it.

    Returns:
        The parsed application configuration, or ``None`` if it could not be parsed.
    """

    pp = _Parser(ctx)
    d = pp.ensure_dict(dct, path)
    return _parse_app_dict(d, path, pp) if d else None


def read_from_path(path: str, ctx: gws.ConfigContext) -> Optional[dict]:
    """Read a configuration file into a dict, without validating it.

    The format is determined by the file extension.

    Args:
        path: Path to the configuration file.
        ctx: Configuration context. Errors are added to it.

    Returns:
        The configuration dict, or ``None`` if the file could not be read or does not contain a dict.
    """
    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return d


##


def _parse_app_dict(dct: dict, path, pp: '_Parser'):
    """Set the time zone, parse the application config and collect all project configs."""
    dct = gws.u.to_dict(dct)
    if not isinstance(dct, dict):
        _register_error(pp.ctx, f'app config must be a dict', path=path)
        return

    # the timezone must be set before everything else
    tz = dct.get('server', {}).get('timeZone', '')
    if tz:
        if gws.lib.datetimex.is_valid_time_zone(tz):
            gws.lib.datetimex.set_local_time_zone(tz)
        else:
            _register_error(pp.ctx, f'invalid time zone: {tz!r}', path=path)
    gws.log.info(f'local time zone="{gws.lib.datetimex.time_zone()}"')

    # remove 'projects' from the config, parse them later on
    inline_projects = dct.pop('projects', [])

    app_cfg = pp.parse_dict(dct, path, as_type='gws.base.application.core.Config')
    if not app_cfg:
        return

    projects = []
    for dcts in inline_projects:
        projects.extend(_parse_projects(dcts, path, pp))

    project_paths = list(app_cfg.get('projectPaths') or [])
    project_dirs = list(app_cfg.get('projectDirs') or [])

    all_project_paths = list(project_paths)
    for dirname in project_dirs:
        all_project_paths.extend(gws.lib.osx.find_files(dirname, CONFIG_PATH_PATTERN, deep=True))

    for pth in sorted(set(all_project_paths)):
        projects.extend(_parse_projects_from_path(pth, pp))

    app_cfg.set('projectPaths', project_paths)
    app_cfg.set('projectDirs', project_dirs)
    app_cfg.set('projects', projects)

    _save_debug(app_cfg, path, '.parsed.json')
    return app_cfg


def _parse_projects_from_path(path, pp: '_Parser'):
    cfg_list = pp.read_from_path(path)
    if not cfg_list:
        return []
    return _parse_projects(cfg_list, path, pp)


def _parse_projects(cfg_list, path, pp: '_Parser'):
    """Parse a project config or a (nested) list of them."""
    ps = []

    for c in _as_flat_list(cfg_list):
        d = pp.ensure_dict(c, path)
        if not d:
            continue
        prj_cfg = pp.parse_dict(d, path, 'gws.ext.config.project')
        if prj_cfg:
            ps.append(prj_cfg)

    return ps


##


class _Parser:
    """Reads configuration files and validates dicts, recording errors in the context."""

    def __init__(self, ctx: gws.ConfigContext):
        """Initialize the context and enable verbose errors.

        Args:
            ctx: Configuration context.
        """
        self.ctx = ctx
        self.ctx.errors = ctx.errors or []
        self.ctx.paths = ctx.paths or set()
        self.ctx.readOptions = ctx.readOptions or set()
        self.ctx.readOptions.add(gws.SpecReadOption.verboseErrors)

    def ensure_dict(self, val, path):
        """Convert a value to a plain dict.

        Args:
            val: Value to convert, a dict or a ``gws.Data`` object.
            path: Path to the configuration file, for error reporting.

        Returns:
            A plain dict, or ``None`` if the value is ``None`` or not a dict.
        """
        if val is None:
            return
        d = _to_plain(val)
        if not isinstance(d, dict):
            _register_error(self.ctx, f'unsupported configuration type {type(val)!r}', path=path)
            return
        return d

    def parse_dict(self, dct: dict, path: str, as_type: str) -> Optional[gws.Config]:
        """Validate a dict against the specs.

        Args:
            dct: Configuration dict.
            path: Path to the configuration file, recorded in the context.
            as_type: Type of the configuration.

        Returns:
            The parsed configuration, or ``None`` if there were errors.
        """
        if not isinstance(dct, dict):
            _register_error(self.ctx, 'unsupported configuration', path=path)
            return
        if path:
            _register_path(self.ctx, path)
        try:
            cfg = self.ctx.specs.read(
                dct,
                as_type,
                path=path,
                options=self.ctx.readOptions,
            )
            return cast(gws.Config, cfg)
        except gws.spec.runtime.ReadError as exc:
            message, _, cei = exc.args
            _register_error(self.ctx, f'parse error: {message}', cei=cei)

    def read_from_path(self, path: str):
        """Read a configuration file and convert the result to plain values.

        Args:
            path: Path to the configuration file.

        Returns:
            The configuration value, or ``None`` if the file could not be read.
        """
        if not os.path.isfile(path):
            _register_error(self.ctx, f'file not found', path=path)
            return

        _register_path(self.ctx, path)
        r = self.read2(path)

        if r:
            r = _to_plain(r)
            _save_debug(r, path, '.src.json')
            return r

    def read2(self, path: str):
        """Read a configuration file using the reader for its extension.

        Args:
            path: Path to the configuration file.

        Returns:
            The configuration value, or ``None`` on errors.
        """
        if path.endswith('.py'):
            return self.read_py(path)
        if path.endswith('.json'):
            return self.read_json(path)
        if path.endswith(('.yml', '.yaml')):
            return self.read_yaml(path)
        if path.endswith('.cx'):
            return self.read_cx(path)

        _register_error(self.ctx, 'unsupported configuration', path=path)

    def read_py(self, path: str):
        """Load a Python configuration file and call its ``main`` function with the context.

        Args:
            path: Path to the configuration file.

        Returns:
            The value returned by ``main``, or ``None`` on errors.
        """
        try:
            fn = gws.lib.dynimport.load_file(path).get('main')
            if not fn:
                _register_error(self.ctx, f'no "main" function found', path=path)
                return
            return fn(self.ctx)
        except Exception as exc:
            gws.log.exception()
            _register_error(self.ctx, f'python error: {exc}', path=path)

    def read_json(self, path: str):
        """Read a JSON configuration file.

        Args:
            path: Path to the configuration file.

        Returns:
            The decoded value, or ``None`` on errors.
        """
        try:
            return gws.lib.jsonx.from_path(path)
        except Exception as exc:
            _register_error(self.ctx, f'json error: {exc}', path=path)

    def read_yaml(self, path: str):
        """Read a YAML configuration file.

        Args:
            path: Path to the configuration file.

        Returns:
            The decoded value, or ``None`` on errors.
        """
        try:
            with open(path, encoding='utf8') as fp:
                return yaml.safe_load(fp)
        except Exception as exc:
            _register_error(self.ctx, f'yaml error: {exc}', path=path)

    def read_cx(self, path: str):
        """Read a ``.cx`` configuration file.

        The file is a ``jump`` template that renders to SLON. Included files are
        recorded in the context. Template variables are ``true``, ``false``,
        ``ctx`` and ``gws``.

        Args:
            path: Path to the configuration file.

        Returns:
            The decoded value, or ``None`` on errors.
        """
        err_cnt = [0]

        def _error_handler(exc, path, line, env):
            _register_syntax_error(self.ctx, path, gws.u.read_file(path), message=repr(exc), line=line)
            err_cnt[0] += 1
            return True

        def _loader(cur_path, load_path):
            if not os.path.isabs(load_path):
                load_path = os.path.abspath(os.path.dirname(cur_path) + '/' + load_path)
            _register_path(self.ctx, load_path)
            return gws.u.read_file(load_path), load_path

        try:
            tpl = gws.lib.vendor.jump.compile_path(path, loader=_loader)
        except gws.lib.vendor.jump.CompileError as exc:
            _register_syntax_error(self.ctx, path, gws.u.read_file(exc.path), message=exc.message, line=exc.line)
            return

        args = {
            'true': True,
            'false': False,
            'ctx': self.ctx,
            'gws': gws,
        }

        slon = gws.lib.vendor.jump.call(tpl, args, error=_error_handler)
        if err_cnt[0] > 0:
            return

        _save_debug(slon, path, '.src.slon')

        try:
            return gws.lib.vendor.slon.loads(slon, as_object=True)
        except gws.lib.vendor.slon.SlonError as exc:
            _register_syntax_error(self.ctx, path, slon, message=exc.args[0], line=exc.args[2])


##


def _register_path(ctx, path):
    ctx.paths.add(path)


def _register_error(ctx: gws.ConfigContext, message: str, **kwargs):
    cei = kwargs.pop('cei', None) or gws.ConfigErrorInfo()
    cei.message = message
    cei.update(kwargs)
    ctx.errors.append(cei)


def _register_warning(ctx: gws.ConfigContext, message: str, **kwargs):
    cei = kwargs.pop('cei', None) or gws.ConfigErrorInfo()
    cei.message = message
    cei.update(kwargs)
    loc = f' in {cei.path!r}' if cei.path else ''
    gws.log.warning(f'CONFIGURATION WARNING: {message}{loc}')
    ctx.warnings.append(cei)


def _register_syntax_error(ctx, path, src, message, line, context=10, cause=None):
    """Add a syntax error with the surrounding source lines to the context."""
    cei = gws.ConfigErrorInfo(
        path=path,
        line=line,
        message=f'syntax error: {message}',
        contextLines=[],
        cause=cause,
    )

    for n, ln in enumerate(src.splitlines(), 1):
        if n < line - context:
            continue
        if n > line + context:
            break
        ln = f'{n}: {ln}'
        if n == line:
            ln = f'>>> {ln}'
        cei.contextLines.append(ln)

    ctx.errors.append(cei)


def _save_debug(src, src_path, ext):
    """Write an intermediate parsing result to the config directory, for debugging."""
    if ext.endswith('.json') and not isinstance(src, str):
        src = gws.lib.jsonx.to_pretty_string(src)
    path = gws.u.write_file(f'{gws.c.CONFIG_DIR}/{gws.u.to_uid(src_path)}{ext}', src)
    return f'saved {path!r}'


def _as_flat_list(ls):
    if not isinstance(ls, (list, tuple)):
        yield ls
    else:
        for x in ls:
            yield from _as_flat_list(x)


def _to_plain(val):
    """Convert ``gws.Data`` objects to dicts recursively, values of keys starting with ``_`` are left as is."""
    if isinstance(val, (list, tuple)):
        return [_to_plain(x) for x in val]
    if isinstance(val, gws.Data):
        val = vars(val)
    if isinstance(val, dict):
        return {k: v if k.startswith('_') else _to_plain(v) for k, v in val.items()}
    return val

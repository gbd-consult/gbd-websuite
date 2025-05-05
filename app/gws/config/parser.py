"""Configuration parser.

Convert configuration files (in different formats) or row config dicts
into ``gws.Config`` objects by validating them against the specs.
"""

from typing import Optional, cast

import os
import yaml

import gws
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.datetimex
import gws.lib.importer
import gws.lib.vendor.jump
import gws.lib.vendor.slon
import gws.spec.runtime

CONFIG_PATH_PATTERN = r'\.(py|json|yaml|cx)$'


def parse_from_path(path: str, as_type: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Parse a configuration from a path.

    Args:
        path: Path to the configuration file.
        as_type: Type of the configuration (e.g., 'gws.base.application.core.Config').
        ctx: Configuration context.
    """

    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return pp.parse_dict(d, path, as_type) if d else None


def parse_dict(dct: dict | gws.Data, path: str, as_type: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Parse a configuration given as python dict.

    Args:
        dct: Dictionary containing the configuration.
        path: Path to the configuration file (for error reporting).
        as_type: Type of the configuration.
        ctx: Configuration context.
    """

    pp = _Parser(ctx)
    d = pp.ensure_dict(dct, path)
    return pp.parse_dict(d, path, as_type) if d else None


def parse_app_from_path(path: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Parse application configuration from a path.

    Args:
        path: Path to the application configuration file.
        ctx: Configuration context.
    """

    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return _parse_app_dict(d, path, pp) if d else None


def parse_app_dict(dct: dict | gws.Data, path: str, ctx: gws.ConfigContext) -> Optional[gws.Config]:
    """Parse application configuration given as python dict.

    Args:
        dct: Dictionary containing the application configuration.
        path: Path to the configuration file (for error reporting).
        ctx: Configuration context.
    """

    pp = _Parser(ctx)
    d = pp.ensure_dict(dct, path)
    return _parse_app_dict(d, path, pp) if d else None


def read_from_path(path: str, ctx: gws.ConfigContext) -> Optional[dict]:
    """Read a configuration file from a path, parse config formats.

    Args:
        path: Path to the configuration file.
        ctx: Configuration context.
    """
    pp = _Parser(ctx)
    val = pp.read_from_path(path)
    d = pp.ensure_dict(val, path)
    return d


##


def _parse_app_dict(dct: dict, path, pp: '_Parser'):
    dct = gws.u.to_dict(dct)
    if not isinstance(dct, dict):
        _register_error(pp.ctx, f'app config must be a dict: {path!r}')
        return

    # the timezone must be set before everything else
    tz = dct.get('server', {}).get('timeZone', '')
    if tz:
        gws.lib.datetimex.set_local_time_zone(tz)
    gws.log.info(f'local time zone="{gws.lib.datetimex.time_zone()}"')

    # remove 'projects' from the config, parse them later on
    inline_projects = dct.pop('projects', [])

    gws.log.info('parsing main configuration...')
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
    def __init__(self, ctx: gws.ConfigContext):
        self.ctx = ctx
        self.ctx.errors = ctx.errors or []
        self.ctx.paths = ctx.paths or set()
        self.ctx.readOptions = ctx.readOptions or set()
        self.ctx.readOptions.add(gws.SpecReadOption.verboseErrors)

    def ensure_dict(self, val, path):
        if val is None:
            return
        d = _to_plain(val)
        if not isinstance(d, dict):
            _register_error(self.ctx, f'unsupported configuration type: {type(val)!r}', path)
            return
        return d

    def parse_dict(self, dct: dict, path: str, as_type: str) -> Optional[gws.Config]:
        if not isinstance(dct, dict):
            _register_error(self.ctx, 'unsupported configuration', path)
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
            message, _, details = exc.args
            lines = []
            pp = details.get('path')
            if pp:
                lines.append(f'PATH: {pp!r}')
            pp = details.get('formatted_value')
            if pp:
                lines.append(f'VALUE: {pp}')
            pp = details.get('formatted_stack')
            if pp:
                lines.extend(pp)
            _register_error(self.ctx, f'parse error: {message}', *lines, cause=exc)

    def read_from_path(self, path: str):
        if not os.path.isfile(path):
            _register_error(self.ctx, f'file not found: {path!r}')
            return

        _register_path(self.ctx, path)
        r = self.read2(path)

        if r:
            r = _to_plain(r)
            _save_debug(r, path, '.src.json')
            return r

    def read2(self, path: str):
        if path.endswith('.py'):
            return self.read_py(path)
        if path.endswith('.json'):
            return self.read_json(path)
        if path.endswith('.yml'):
            return self.read_yaml(path)
        if path.endswith('.cx'):
            return self.read_cx(path)

        _register_error(self.ctx, 'unsupported configuration', path)

    def read_py(self, path: str):
        try:
            fn = gws.u.require(gws.lib.importer.load_file(path).get('main'))
            return fn(self.ctx)
        except Exception as exc:
            gws.log.exception()
            _register_error(self.ctx, f'python error: {exc}', cause=exc)

    def read_json(self, path: str):
        try:
            return gws.lib.jsonx.from_path(path)
        except Exception as exc:
            _register_error(self.ctx, 'json error', cause=exc)

    def read_yaml(self, path: str):
        try:
            with open(path, encoding='utf8') as fp:
                return yaml.safe_load(fp)
        except Exception as exc:
            _register_error(self.ctx, 'yaml error', cause=exc)

    def read_cx(self, path: str):
        err_cnt = [0]

        def _error_handler(exc, path, line, env):
            _register_syntax_error(self.ctx, path, gws.u.read_file(path), repr(exc), line)
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
            _register_syntax_error(self.ctx, path, gws.u.read_file(exc.path), exc.message, exc.line, cause=exc)
            return

        args = args = {
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
            _register_syntax_error(self.ctx, path, slon, exc.args[0], exc.args[2], cause=exc)


##


def _register_path(ctx, path):
    ctx.paths.add(path)


def _register_error(ctx, message, *args, cause=None):
    err = gws.ConfigurationError(message, *args)
    if cause:
        err.__cause__ = cause
    ctx.errors.append(err)


def _register_syntax_error(ctx, path, src, message, line, context=10, cause=None):
    lines = [f'PATH: {path!r}']

    for n, ln in enumerate(src.splitlines(), 1):
        if n < line - context:
            continue
        if n > line + context:
            break
        ln = f'{n}: {ln}'
        if n == line:
            ln = f'>>> {ln}'
        lines.append(ln)

    _register_error(ctx, f'syntax error: {message}', *lines, cause=cause)


def _save_debug(src, src_path, ext):
    if ext.endswith('.json') and not isinstance(src, str):
        src = gws.lib.jsonx.to_pretty_string(src)
    gws.u.write_file(f'{gws.c.CONFIG_DIR}/{gws.u.to_uid(src_path)}{ext}', src)


def _as_flat_list(ls):
    if not isinstance(ls, (list, tuple)):
        yield ls
    else:
        for x in ls:
            yield from _as_flat_list(x)


def _to_plain(val):
    if isinstance(val, (list, tuple)):
        return [_to_plain(x) for x in val]
    if isinstance(val, gws.Data):
        val = vars(val)
    if isinstance(val, dict):
        return {k: v if k.startswith('_') else _to_plain(v) for k, v in val.items()}
    return val

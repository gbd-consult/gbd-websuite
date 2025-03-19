"""Parse and validate the main cfg and project configs"""

from typing import Optional, Any

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


class ParseResult(gws.Data):
    """Result of a config parsing"""

    config: gws.Config
    errors: list[gws.Error]
    paths: set[str]


def parse_path(path: str, as_type: str, specs: gws.SpecRuntime, read_options=None) -> ParseResult:
    """Parse configuration from a path."""

    pp = _Parser(specs, read_options)
    pp.parse_path(path, as_type)
    return pp.result


def parse_config(cfg: Any, as_type: str, specs: gws.SpecRuntime, path='', read_options=None) -> ParseResult:
    """Parse configuration given as python dict or list."""

    pp = _Parser(specs, read_options)
    pp.parse_payload(_to_plain_type(cfg), path, as_type)
    return pp.result


def parse_app_config_path(path, specs: gws.SpecRuntime) -> ParseResult:
    pp = _Parser(specs, None)
    payload = pp.read(path)
    if payload:
        _parse_app_config(payload, path, pp)
    return pp.result


def parse_app_config(cfg, specs, path) -> ParseResult:
    pp = _Parser(specs, None)
    _parse_app_config(_to_plain_type(cfg), path, pp)
    return pp.result


def _parse_app_config(cfg, path, pp: '_Parser'):
    if not isinstance(cfg, dict):
        pp.result.errors.append(_error(f'app config must be a dict: {path!r}'))
        return

    # the timezone must be set before everything else
    tz = cfg.get('server', {}).get('timeZone', '')
    if tz:
        gws.lib.datetimex.set_local_time_zone(tz)
    gws.log.info(f'local time zone is "{gws.lib.datetimex.time_zone()}"')

    # remove 'projects' from the config, parse them later on
    inline_projects = cfg.pop('projects', [])

    gws.log.info('parsing main configuration...')
    pp.parse_payload(cfg, path, as_type='gws.base.application.core.Config')
    if not pp.result.config:
        return

    app_cfg = pp.result.config

    # find projects from 'paths' and 'dirs'

    pp.result.config.projectPaths = pp.result.config.projectPaths or []
    pp.result.config.projectDirs = pp.result.config.projectDirs or []

    project_paths = list(pp.result.config.projectPaths)
    for dirname in pp.result.config.projectDirs:
        project_paths.extend(gws.lib.osx.find_files(dirname, CONFIG_PATH_PATTERN, deep=True))

    app_cfg.projects = []

    for cfg in inline_projects:
        app_cfg.projects.extend(_parse_projects(cfg, path, pp))

    for pth in sorted(set(project_paths)):
        app_cfg.projects.extend(_parse_projects(None, pth, pp))

    save_debug(app_cfg, path, '.parsed.json')


def _parse_projects(cfg, path, pp: '_Parser'):
    if not cfg:
        pp2 = _Parser(pp.specs)
        cfg = pp2.read(path)
        pp.add_errors_and_paths(pp2)

    if not cfg:
        return []

    ps = []

    for c in _as_flat_list(cfg):
        pp2 = _Parser(pp.specs)
        pp2.parse_payload(c, path, 'gws.ext.config.project')
        pp.add_errors_and_paths(pp2)
        if pp2.result.config:
            ps.append(pp2.result.config)

    return ps


##

class _Parser:
    def __init__(self, specs: gws.SpecRuntime, read_options=None):
        self.result = ParseResult(config=None, errors=[], paths=set())
        self.specs = specs
        self.read_options = read_options

    def add_errors_and_paths(self, other: '_Parser'):
        self.result.errors.extend(other.result.errors)
        self.result.paths.update(other.result.paths)

    def parse_path(self, path, as_type):
        payload = self.read(path)
        if payload:
            self.parse_payload(payload, path, as_type)

    def parse_payload(self, payload, path, as_type):
        if path:
            self.result.paths.add(path)
        try:
            read_options = self.read_options or set()
            read_options.add(gws.SpecReadOption.verboseErrors)
            self.result.config = self.specs.read(payload, as_type, path=path, options=read_options)
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
            self.result.errors.append(_error(f'parse error: {message}', *lines, cause=exc))

    def read(self, path: str):
        if not os.path.isfile(path):
            self.result.errors.append(_error(f'file not found: {path!r}'))
            return

        self.result.paths.add(path)
        payload = self.read2(path)

        if payload:
            payload = _to_plain_type(payload)
            save_debug(payload, path, '.src.json')
            return payload

    def read2(self, path: str):
        if path.endswith('.py'):
            return self.read_py(path)
        if path.endswith('.json'):
            return self.read_json(path)
        if path.endswith('.yml'):
            return self.read_yaml(path)
        if path.endswith('.cx'):
            return self.read_cx(path)

        self.result.errors.append(_error('unsupported config format', path))

    def read_py(self, path: str):
        try:
            fn = gws.lib.importer.load_file(path).get('main')
            return fn()
        except Exception as exc:
            self.result.errors.append(_error('python error', cause=exc))

    def read_json(self, path: str):
        try:
            return gws.lib.jsonx.from_path(path)
        except Exception as exc:
            self.result.errors.append(_error('json error', cause=exc))

    def read_yaml(self, path: str):
        try:
            with open(path, encoding='utf8') as fp:
                return yaml.safe_load(fp)
        except Exception as exc:
            self.result.errors.append(_error('yaml error', cause=exc))

    def read_cx(self, path: str):
        runtime_errors = []

        def _error_handler(exc, path, line, env):
            runtime_errors.append(_syntax_error(path, gws.u.read_file(path), repr(exc), line))
            return True

        def _loader(cur_path, load_path):
            if not os.path.isabs(load_path):
                load_path = os.path.abspath(os.path.dirname(cur_path) + '/' + load_path)
            self.result.paths.add(load_path)
            return gws.u.read_file(load_path), load_path

        try:
            tpl = gws.lib.vendor.jump.compile_path(path, loader=_loader)
        except gws.lib.vendor.jump.CompileError as exc:
            self.result.errors.append(
                _syntax_error(path, gws.u.read_file(exc.path), exc.message, exc.line, cause=exc))
            return

        src = gws.lib.vendor.jump.call(tpl, args={'true': True, 'false': False}, error=_error_handler)
        if runtime_errors:
            self.result.errors.extend(runtime_errors)
            return

        save_debug(src, path, '.src.slon')

        try:
            return gws.lib.vendor.slon.loads(src, as_object=True)
        except gws.lib.vendor.slon.SlonError as exc:
            self.result.errors.append(_syntax_error(path, src, exc.args[0], exc.args[2], cause=exc))


##

def save_debug(src, src_path, ext):
    if ext.endswith('.json') and not isinstance(src, str):
        src = gws.lib.jsonx.to_pretty_string(src)
    gws.u.write_file(f"{gws.c.CONFIG_DIR}/{gws.u.to_uid(src_path)}{ext}", src)


def _syntax_error(path, src, message, line, context=10, cause=None):
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

    return _error(f'syntax error: {message}', *lines, cause=cause)


def _as_flat_list(ls):
    if not isinstance(ls, (list, tuple)):
        yield ls
    else:
        for x in ls:
            yield from _as_flat_list(x)


def _to_plain_type(val):
    if isinstance(val, list):
        return [_to_plain_type(x) for x in val]
    if isinstance(val, tuple):
        return tuple(_to_plain_type(x) for x in val)
    if isinstance(val, gws.Data):
        val = vars(val)
    if isinstance(val, dict):
        return {k: _to_plain_type(v) for k, v in val.items()}
    return val


def _error(message, *args, cause=None):
    err = gws.ConfigurationError(message, *args)
    if cause:
        err.__cause__ = cause
    return err

"""Parse and validate the main cfg and project configs"""

import os
import yaml

import gws
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.importer
import gws.lib.vendor.jump
import gws.lib.vendor.slon
import gws.spec.runtime

import gws.types as t

CONFIG_PATH_PATTERN = r'\bconfig\.(py|json|yaml|cx)$'
CONFIG_FUNCTION_NAME = 'config'

def parse(specs: gws.ISpecRuntime, value, type_name: str, source_path='', read_options=None):
    """Parse a dictionary according to the klass spec and return a config (Data) object"""

    try:
        read_options = read_options or set()
        read_options.add(gws.SpecReadOption.verboseErrors)
        return specs.read(value, type_name, path=source_path, options=read_options)
    except gws.spec.runtime.ReadError as exc:
        message, _, details = exc.args
        lines = []
        p = details.get('path')
        if p:
            lines.append(f'PATH: {p!r}')
        p = details.get('formatted_value')
        if p:
            lines.append(f'VALUE: {p}')
        p = details.get('formatted_stack')
        if p:
            lines.extend(p)
        raise gws.ConfigurationError(f'parse error: {message}', lines) from exc


class ConfigParser:
    """Read and parse the main config file"""

    def __init__(self, specs: gws.ISpecRuntime):
        self.specs = specs
        self.errors = []
        self.paths = set()

    def parse_main(self, config_path=None) -> t.Optional[gws.Config]:
        payload = self.read(config_path)
        if not payload:
            return None
        return self.parse_main_from_dict(payload, config_path)

    def parse_main_from_dict(self, dct, config_path) -> t.Optional[gws.Config]:
        prj_dicts = []

        for prj_cfg in dct.pop('projects', []):
            for prj_dict in _as_flat_list(prj_cfg):
                prj_dicts.append([prj_dict, config_path])

        gws.log.info('parsing main configuration...')
        try:
            app_cfg = parse(self.specs, dct, 'gws.base.application.core.Config', config_path)
        except gws.ConfigurationError as exc:
            self.errors.append(exc)
            return None

        app_cfg.projectPaths = app_cfg.projectPaths or []
        app_cfg.projectDirs = app_cfg.projectDirs or []

        prj_paths = list(app_cfg.projectPaths)
        for dirname in app_cfg.projectDirs:
            prj_paths.extend(gws.lib.osx.find_files(dirname, CONFIG_PATH_PATTERN))

        for prj_path in sorted(set(prj_paths)):
            payload = self.read(prj_path)
            if not payload:
                continue
            for prj_dict in _as_flat_list(payload):
                prj_dicts.append([prj_dict, prj_path])

        app_cfg.projects = []

        for prj_dict, prj_path in prj_dicts:
            uid = prj_dict.get('uid') or prj_dict.get('title') or '?'
            gws.log.info(f'parsing project {uid!r}...')
            try:
                prj_cfg = parse(self.specs, prj_dict, 'gws.ext.config.project', prj_path)
            except gws.ConfigurationError as exc:
                self.errors.append(exc)
                continue
            app_cfg.projects.append(prj_cfg)

        app_cfg.configPaths = sorted(self.paths)
        return app_cfg

    def read(self, path):
        if not os.path.isfile(path):
            self.errors.append(_error(f'file not found: {path!r}'))
            return
        payload = self.read2(path)
        if payload:
            _save_intermediate(path, gws.lib.jsonx.to_pretty_string(payload), 'json')
            return payload

    def read2(self, path):
        if path.endswith('.py'):
            try:
                mod = gws.lib.importer.import_from_path(path)
                fn = getattr(mod, CONFIG_FUNCTION_NAME)
                payload = fn()
            except Exception as exc:
                self.errors.append(_error('python error', cause=exc))
                return
            if not isinstance(payload, dict):
                payload = _as_dict(payload)
            self.paths.add(path)
            return payload

        if path.endswith('.json'):
            try:
                payload = gws.lib.jsonx.from_path(path)
            except Exception as exc:
                self.errors.append(_error('json error', cause=exc))
                return
            self.paths.add(path)
            return payload

        if path.endswith('.yaml'):
            try:
                with open(path, encoding='utf8') as fp:
                    payload = yaml.safe_load(fp)
            except Exception as exc:
                self.errors.append(_error('yaml error', cause=exc))
                return
            self.paths.add(path)
            return payload

        if path.endswith('.cx'):
            return self.parse_cx_config(path)

        self.errors.append(_error('unsupported config format', path))

    def parse_cx_config(self, path):
        paths = {path}
        runtime_errors = []

        def _error_handler(exc, path, line, env):
            runtime_errors.append(_syntax_error(path, gws.read_file(path), repr(exc), line))
            return True

        def _loader(cur_path, p):
            if not os.path.isabs(p):
                d = os.path.dirname(cur_path)
                p = os.path.abspath(os.path.join(d, p))
            paths.add(p)
            return gws.read_file(p), p

        try:
            tpl = gws.lib.vendor.jump.compile_path(path, loader=_loader)
        except gws.lib.vendor.jump.CompileError as exc:
            self.errors.append(
                _syntax_error(path, gws.read_file(exc.path), exc.message, exc.line, cause=exc))
            return

        src = gws.lib.vendor.jump.call(tpl, args={'true': True, 'false': False}, error=_error_handler)
        if runtime_errors:
            self.errors.extend(runtime_errors)
            return

        _save_intermediate(path, src, 'slon')

        try:
            payload = gws.lib.vendor.slon.loads(src, as_object=True)
        except gws.lib.vendor.slon.SlonError as exc:
            self.errors.append(_syntax_error(path, src, exc.args[0], exc.args[2], cause=exc))
            return

        self.paths.update(paths)
        return payload


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


def _save_intermediate(path, txt, ext):
    gws.write_file(f"{gws.CONFIG_DIR}/{gws.to_uid(path)}.parsed.{ext}", txt)


def _as_flat_list(ls):
    if not isinstance(ls, (list, tuple)):
        yield ls
    else:
        for x in ls:
            yield from _as_flat_list(x)


def _as_dict(val):
    if isinstance(val, list):
        return [_as_dict(x) for x in val]
    if isinstance(val, tuple):
        return tuple(_as_dict(x) for x in val)
    if isinstance(val, gws.Data):
        val = vars(val)
    if isinstance(val, dict):
        return {k: _as_dict(v) for k, v in val.items()}
    return val


def _error(message, *args, cause=None):
    err = gws.ConfigurationError(message, *args)
    if cause:
        err.__cause__ = cause
    return err

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

CONFIG_PATH_PATTERN = r'\bconfig\.(py|json|yaml|cx)$'
CONFIG_FUNCTION_NAME = 'config'

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]


def parse(specs: gws.ISpecRuntime, value, type_name: str, source_path='', read_options=None):
    """Parse a dictionary according to the klass spec and return a config (Data) object"""

    try:
        read_options = read_options or set()
        read_options.add('verbose_errors')
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


def real_config_path(config_path=None):
    p = config_path or os.getenv('GWS_CONFIG')
    if p:
        return p
    for p in DEFAULT_CONFIG_PATHS:
        if gws.is_file(p):
            return p


def parse_main(specs: gws.ISpecRuntime, config_path=None) -> gws.Config:
    """Read and parse the main config file"""

    config_path = real_config_path(config_path)
    if not config_path:
        raise gws.ConfigurationError('no configuration file found')
    gws.log.info(f'using config {config_path!r}...')
    dct, related_paths = _read(config_path)
    return parse_main_from_dict(specs, dct, config_path, related_paths)


def parse_main_from_dict(specs: gws.ISpecRuntime, dct, config_path, related_paths) -> gws.Config:
    prj_dicts = []

    for prj_cfg in dct.pop('projects', []):
        for prj_dict in _as_flat_list(prj_cfg):
            prj_dicts.append([prj_dict, config_path])

    gws.log.info('parsing main configuration...')
    app_cfg = parse(specs, dct, 'gws.base.application.Config', config_path)

    app_cfg.configPaths = related_paths
    app_cfg.projectPaths = app_cfg.projectPaths or []
    app_cfg.projectDirs = app_cfg.projectDirs or []

    prj_paths = list(app_cfg.projectPaths)
    for dirname in app_cfg.projectDirs:
        prj_paths.extend(gws.lib.osx.find_files(dirname, CONFIG_PATH_PATTERN))

    for prj_path in sorted(set(prj_paths)):
        prj_cfg, paths = _read(prj_path)
        app_cfg.configPaths.extend(paths)
        for prj_dict in _as_flat_list(prj_cfg):
            prj_dicts.append([prj_dict, prj_path])

    app_cfg.projects = []

    for prj_dict, prj_path in prj_dicts:
        uid = prj_dict.get('uid') or prj_dict.get('title') or '?'
        gws.log.info(f'parsing project {uid!r}...')
        app_cfg.projects.append(parse(specs, prj_dict, 'gws.ext.config.project', prj_path))

    return app_cfg


def _read(path):
    if not os.path.isfile(path):
        raise gws.ConfigurationError(f'file not found: {path!r}')
    try:
        dct, paths = _read2(path)
    except gws.ConfigurationError:
        raise
    except Exception as exc:
        raise gws.ConfigurationError(f'read error: {path!r}') from exc

    _save_intermediate(path, gws.lib.jsonx.to_pretty_string(dct), 'json')
    return dct, paths


def _read2(path):
    if path.endswith('.py'):
        mod = gws.lib.importer.import_from_path(path)
        fn = getattr(mod, CONFIG_FUNCTION_NAME)
        dct = fn()
        if not isinstance(dct, dict):
            dct = _as_dict(dct)
        return dct, [path]

    if path.endswith('.json'):
        return gws.lib.jsonx.from_path(path), [path]

    if path.endswith('.yaml'):
        with open(path, encoding='utf8') as fp:
            dct = yaml.safe_load(fp)
        return dct, [path]

    if path.endswith('.cx'):
        return _parse_cx_config(path)


def _parse_cx_config(path):
    paths = {path}
    runtime_errors = []

    def _error_handler(exc, path, line, env):
        runtime_errors.append(_syntax_error(path, gws.read_file(path), repr(exc), line))

    def _loader(cur_path, p):
        if not os.path.isabs(p):
            d = os.path.dirname(cur_path)
            p = os.path.abspath(os.path.join(d, p))
        paths.add(p)
        return gws.read_file(p), p

    try:
        tpl = gws.lib.vendor.jump.compile_path(path, loader=_loader)
    except gws.lib.vendor.jump.CompileError as exc:
        raise _syntax_error(path, gws.read_file(exc.path), exc.message, exc.line) from exc

    src = gws.lib.vendor.jump.call(tpl, args={'true': True, 'false': False}, error=_error_handler)

    if runtime_errors:
        raise runtime_errors[0]

    _save_intermediate(path, src, 'slon')

    try:
        dct = gws.lib.vendor.slon.loads(src, as_object=True)
    except gws.lib.vendor.slon.SlonError as exc:
        raise _syntax_error(path, src, exc.args[0], exc.args[2]) from exc

    return dct, list(paths)


def _syntax_error(path, src, message, line, context=10):
    lines = [f'PATH: {path!r}']

    for n, t in enumerate(src.splitlines(), 1):
        if n < line - context:
            continue
        if n > line + context:
            break
        t = str(n) + ': ' + t
        if n == line:
            t = '>>>' + t
        lines.append(t)

    return gws.ConfigurationError(f'syntax error: {message}', lines)


def _save_intermediate(path, txt, ext):
    p = gws.lib.osx.parse_path(path)
    gws.write_file(f"{gws.CONFIG_DIR}/{p['name']}.parsed.{ext}", txt)


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

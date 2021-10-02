"""Parse and validate the main cfg and project configs"""

import os

import yaml

import gws
import gws.lib.json2
import gws.lib.os2
import gws.lib.vendor.chartreux as chartreux
import gws.lib.vendor.slon as slon
import gws.spec.runtime

CONFIG_PATH_PATTERN = r'\bconfig\.(py|json|yaml|cx)$'
CONFIG_FUNCTION_NAME = 'config'

DEFAULT_CONFIG_PATHS = [
    '/data/config.cx',
    '/data/config.json',
    '/data/config.yaml',
    '/data/config.py',
]


def parse(specs: gws.ISpecRuntime, value, type_name: str, source_path='', with_internal_objects=False):
    """Parse a dictionary according to the klass spec and return a config (Data) object"""

    try:
        return specs.read_value(value, type_name, source_path, with_strict_mode=True, with_error_details=True, with_internal_objects=with_internal_objects)
    except gws.spec.runtime.Error as e:
        code, msg, _, details = e.args
        raise gws.ConfigurationError(
            code + ': ' + msg,
            details.get('path'),
            details.get('formatted_value'),
            details.get('formatted_stack'))


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
    dct, paths = _read(config_path)
    return parse_main_from_dict(specs, dct, paths)


def parse_main_from_dict(specs: gws.ISpecRuntime, dct, config_paths) -> gws.Config:
    config_path = config_paths[0]
    prj_dicts = []

    for prj_cfg in dct.pop('projects', []):
        for prj_dict in _as_flat_list(prj_cfg):
            prj_dicts.append([prj_dict, config_path])

    gws.log.info('parsing main configuration...')
    app_cfg = parse(specs, dct, 'gws.base.application.Config', config_path)

    app_cfg.configPaths = config_paths
    app_cfg.projectPaths = app_cfg.projectPaths or []
    app_cfg.projectDirs = app_cfg.projectDirs or []

    prj_paths = list(app_cfg.projectPaths)
    for dirname in app_cfg.projectDirs:
        prj_paths.extend(gws.lib.os2.find_files(dirname, CONFIG_PATH_PATTERN))

    for prj_path in sorted(set(prj_paths)):
        prj_cfg, paths = _read(prj_path)
        config_paths.extend(paths)
        for prj_dict in _as_flat_list(prj_cfg):
            prj_dicts.append([prj_dict, prj_path])

    app_cfg.projects = []

    for prj_dict, prj_path in prj_dicts:
        uid = prj_dict.get('uid') or prj_dict.get('title') or '???'
        gws.log.info(f'parsing project {uid!r}...')
        app_cfg.projects.append(parse(specs, prj_dict, 'gws.base.project.core.Config', prj_path))

    return app_cfg


def _read(path):
    if not os.path.isfile(path):
        raise gws.ConfigurationError('file not found', path, '', None)
    try:
        dct, paths = _read2(path)
    except gws.ConfigurationError:
        raise
    except Exception as exc:
        raise gws.ConfigurationError('read error: %s' % e, path, '', None) from exc

    _save_intermediate(path, gws.lib.json2.to_pretty_string(dct), 'json')
    return dct, paths


def _read2(path):
    if path.endswith('.py'):
        mod_name = 'gws.cfg.' + gws.as_uid(path)
        mod = gws.import_from_path(path, mod_name)
        fn = getattr(mod, CONFIG_FUNCTION_NAME)
        dct = fn()
        if not isinstance(dct, dict):
            dct = _as_dict(dct)
        return dct, [path]

    if path.endswith('.json'):
        return gws.lib.json2.from_path(path), [path]

    if path.endswith('.yaml'):
        with open(path, encoding='utf8') as fp:
            dct = yaml.load(fp)
        return dct, [path]

    if path.endswith('.cx'):
        return _parse_cx_config(path)


def _parse_cx_config(path):
    paths = {path}
    runtime_exc = []

    def _err(exc, path, line):
        if not runtime_exc:
            runtime_exc.append(_syntax_error(path, gws.read_file(path), repr(exc), line))

    def _finder(cur_path, p):
        if not os.path.isabs(p):
            d = os.path.dirname(cur_path)
            p = os.path.abspath(os.path.join(d, p))
        paths.add(p)
        return p

    try:
        tpl = chartreux.compile_path(
            path,
            syntax={'start': '{{', 'end': '}}'},
            finder=_finder
        )
    except chartreux.compiler.Error as e:
        raise _syntax_error(path, gws.read_file(e.path), e.message, e.line)

    src = chartreux.call(
        tpl,
        context={'true': True, 'false': False},
        error=_err)

    if runtime_exc:
        raise runtime_exc[0]

    _save_intermediate(path, src, 'slon')

    try:
        dct = slon.loads(src, as_object=True)
    except slon.SlonError as e:
        raise _syntax_error(path, src, e.args[0], e.args[2])

    return dct, list(paths)


def _syntax_error(path, src, message, line, context=10):
    lines = []

    for n, t in enumerate(src.splitlines(), 1):
        if n < line - context:
            continue
        if n > line + context:
            break
        t = str(n) + ': ' + t
        if n == line:
            t = '>>>' + t
        lines.append(t)

    return gws.ConfigurationError(message, path, '\n'.join(lines), None)


def _save_intermediate(path, txt, ext):
    p = gws.lib.os2.parse_path(path)
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

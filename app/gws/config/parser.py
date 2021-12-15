"""Parse and validate the main cfg and project configs"""

import os

import yaml

import gws
import gws.core.spec
import gws.tools.misc
import gws.tools.os2
import gws.tools.json2
import gws.tools.vendor.chartreux as chartreux
import gws.tools.vendor.slon as slon

import gws.types as t

from . import error, spec

config_path_pattern = r'\bconfig\.(py|json|yaml|cx)$'
config_function_name = 'config'


def parse(dct, type_name, source_path='', strict=True, with_internal_objects=False):
    """Parse a dictionary according to the klass spec and return a config (Data) object"""

    try:
        return spec.validator().read_value(dct, type_name, source_path, strict=strict, with_internal_objects=with_internal_objects)
    except gws.core.spec.Error as e:
        raise error.ParseError(*e.args)


def parse_main(path):
    """Read and parse the main config file"""

    dct, cfg_paths = _read(path)

    prj_configs = []

    for prj_cfg in dct.pop('projects', []):
        for pc in _as_flat_list(prj_cfg):
            prj_configs.append([pc, path])

    gws.log.info('parsing main configuration...')
    app = parse(dct, 'gws.common.application.Config', path)

    app.configPaths = cfg_paths
    app.projectPaths = app.projectPaths or []
    app.projectDirs = app.projectDirs or []

    prj_paths = app.projectPaths
    for dirname in app.projectDirs:
        prj_paths.extend(gws.tools.os2.find_files(dirname, config_path_pattern))

    for prj_path in sorted(set(prj_paths)):
        prj_cfg, prj_cfg_paths = _read(prj_path)
        cfg_paths.extend(prj_cfg_paths)
        for pc in _as_flat_list(prj_cfg):
            prj_configs.append([pc, prj_path])

    app.projects = []

    for pc, prj_path in prj_configs:
        uid = pc.get('uid') or pc.get('title') or '???'
        gws.log.info(f'parsing project {uid!r}...')
        app.projects.append(parse(pc, 'gws.common.project.Config', prj_path))

    return app


def _read(path):
    if not os.path.isfile(path):
        raise error.ParseError('file not found', path, '', None)
    try:
        dct, paths = _read2(path)
    except error.ParseError:
        raise
    except Exception as e:
        raise error.ParseError('read error: %s' % e, path, '', None) from e

    _save_intermediate(path, gws.tools.json2.to_pretty_string(dct), 'json')
    return dct, paths


def _read2(path):
    if path.endswith('.py'):
        mod_name = 'gws.cfg.' + gws.as_uid(path)
        mod = gws.tools.misc.load_source(path, mod_name)
        fn = getattr(mod, config_function_name)
        dct = fn()
        if not isinstance(dct, dict):
            dct = _as_dict(dct)
        return dct, [path]

    if path.endswith('.json'):
        return gws.tools.json2.from_path(path), [path]

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

    return error.ParseError(message, path, '\n'.join(lines), None)


def _save_intermediate(path, txt, ext):
    p = gws.tools.os2.parse_path(path)
    d = gws.VAR_DIR + '/config'
    gws.write_file(f"{d}/{p['name']}.parsed.{ext}", txt)


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
    if isinstance(val, t.Data):
        val = vars(val)
    if isinstance(val, dict):
        return {k: _as_dict(v) for k, v in val.items()}
    return val

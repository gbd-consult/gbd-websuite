"""Parse and validate the main cfg and project configs"""

import json
import os
import re

import yaml

import gws
import gws.types
import gws.types.spec
import gws.tools.misc as misc

from . import error

config_path_pattern = r'\bconfig\.(py|json|yaml)$'
config_function_name = 'config'


def parse(dct, klass, path=''):
    """Parse a dictionary according to the klass spec and return a config (Data) object"""

    try:
        return _validator().get(dct, klass, path)
    except gws.types.spec.Error as e:
        raise error.ParseError(*e.args)


def parse_main(path):
    """Read and parse the main config file"""

    dct = _read(path)

    prj_configs = []
    for pc in dct.pop('projects', []):
        prj_configs.append([pc, path])

    cfg = parse(dct, 'gws.common.application.Config', path)

    prj_paths = cfg.get('projectPaths') or []
    for dirname in cfg.get('projectDirs') or []:
        prj_paths.extend(misc.find_files(dirname, config_path_pattern))

    for prj_path in sorted(set(prj_paths)):
        pcs = _read(prj_path)
        if not isinstance(pcs, list):
            pcs = [pcs]
        for pc in pcs:
            prj_configs.append([pc, prj_path])

    cfg.projects = []

    for pc, prj_path in prj_configs:
        if pc.get('multi'):
            cfg.projects.extend(_parse_multi_project(pc, prj_path))
        else:
            cfg.projects.append(parse(pc, 'gws.common.project.Config', path))

    return cfg


def _read(path):
    if not os.path.exists(path):
        raise error.ParseError('file not found', path, '', '')
    try:
        return _read2(path)
    except Exception as e:
        raise error.ParseError('read error: %s' % e, path, '', '') from e


def _read2(path):
    if path.endswith('.py'):
        mod_name = 'gws.cfg.' + gws.as_uid(path)
        mod = misc.load_source(path, mod_name)
        fn = getattr(mod, config_function_name)
        return fn()

    if path.endswith('.json'):
        with open(path, encoding='utf8') as fp:
            return json.load(fp)

    if path.endswith('.yaml'):
        with open(path, encoding='utf8') as fp:
            return yaml.load(fp)


def load_spec(kind, lang=None):
    path = f'{gws.APP_DIR}/spec/gen/{kind}.spec.json'

    if lang and lang != 'en':
        p = f'{gws.APP_DIR}/spec/gen/{lang}.{kind}.spec.json'
        if os.path.exists(p):
            path = p

    with open(path) as fp:
        return json.load(fp)


def _init_validator():
    spec = load_spec('config')
    return gws.types.spec.Validator(spec['types'], strict=True)


def _validator():
    return gws.get_global('config_validator', _init_validator)


def _parse_multi_project(cfg, path):
    mm = cfg.pop('multi')
    pfx = re.match(r'^/([\w-]+/)*', mm)
    if pfx:
        # absolute multi match like /foo/bar.*/baz? - try to extract the base path
        dirname = pfx.group(0)
        if dirname == '/':
            raise error.ParseError('"multiMatch" cannot be root', path, '', mm)
    else:
        dirname = os.path.dirname(path)

    if not os.path.isdir(dirname):
        raise error.ParseError('"multiMatch" directory {dirname!} not found', path, '', mm)

    res = []

    for p in misc.find_files(dirname, mm):
        gws.log.info(f'multi-project: found {p!r}')
        m = re.search(mm, p)
        args = {'$' + str(n): s for n, s in enumerate(m.groups(), 1)}
        args.update(misc.parse_path(p))
        dct = _deep_format(cfg, args)
        res.append(parse(dct, 'gws.common.project.Config', path))

    if not res:
        gws.log.warn(f'no files found for the multi project "{path}"')

    return res


def _deep_format(x, a):
    if isinstance(x, dict):
        return {k: _deep_format(v, a) for k, v in x.items()}
    if isinstance(x, list):
        return [_deep_format(v, a) for v in x]
    if isinstance(x, str):
        return misc.format_placeholders(x, a)
    return x



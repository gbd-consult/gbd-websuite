"""Combine demos from the source tree into a single config."""

import os
import sys
import re
import json

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../app')
sys.path.append(APP_DIR)

import gws.lib.cli as cli
import gws.lib.vendor.jump
import gws.lib.vendor.slon

USAGE = """
GWS Demo Compiler
~~~~~~~~~~~~~~~~~
  
    python3 demo.py <options>

Options:
    -include <path>
        include a file in the config

    -only <pattern>
        a regex pattern to use only specific demos

    -out <path>
        target json path
"""


def main(args):
    all_paths = list(cli.find_files(APP_DIR, r'_demo/.+?'))

    config = ''

    # basic demo library - from this dir

    for path in sorted(cli.find_files(CUR_DIR, r'\.cx$')):
        config += read_config_file(path)

    # extra include

    if args.get('include'):
        config += read_config_file(args.get('include'))

    # template init code

    config += '@demo_init\n'

    # include projects

    only = args.get('only') or '.'
    for path in all_paths:
        if path.endswith('.cx') and re.search(only, path):
            config += read_config_file(path)

    # render everything

    try:
        config = gws.lib.vendor.jump.render(config)
    except gws.lib.vendor.jump.CompileError as exc:
        error_lines(config, exc.line)
        cli.fatal(str(exc))
    try:
        config = gws.lib.vendor.slon.parse(config, as_object=True)
    except gws.lib.vendor.slon.SlonError as exc:
        error_lines(config, exc.args[2])
        cli.fatal(str(exc))

    # all done!

    res = json.dumps(config, indent=4, ensure_ascii=False)
    if args.get('out'):
        cli.write_file(args.get('out'), res)
    else:
        print(res)


def error_lines(src, err):
    lines = src.split('\n')
    maxl = len(str(len(lines)))
    for n, s in enumerate(lines, 1):
        if n < err - 20:
            continue
        if n > err + 20:
            break
        s = f'{n:{maxl}d}:  {s}'
        if n == err:
            s = '>>> ' + s
        cli.error(s)


START_PROJECT = 'projects+ {'

RE_ASSET = r'''(?x)
    ([\x22\x27])
    (
        [\w./-]+?
        \.
        (?:
            html|qgs|json|png|tiff
        )
    )
    \1
'''


def read_config_file(path):
    text = cli.read_file(path)
    text = generate_project_id(text, path)
    text = relocate_assets(text, path)
    return text + '\n\n'


def generate_project_id(text, path):
    # if this is a project config, inject an uid generated from its path

    if START_PROJECT not in text:
        return text

    m = re.search(r'title (["\'])(.+?)\1', text)
    if not m:
        return text

    uid = re.sub(r'\W+', '_', m.group(2).strip().lower())
    path = path.replace(APP_DIR, '')
    extra = f"""
        uid "{uid}"
        metadata.authorityIdentifier "/app/{path}"
    """

    return text.replace(START_PROJECT, START_PROJECT + extra)


def relocate_assets(text, path):
    # replace asset paths with absolute paths

    return re.sub(RE_ASSET, lambda m: relocate_asset(m, path), text)


def relocate_asset(m, path):
    quot, name = m.groups()
    ps = os.path.realpath(os.path.join(os.path.dirname(path), name))
    if not ps.startswith(APP_DIR):
        return m.group(0)
    pd = ps.replace(APP_DIR, '/gws-app')
    # print(f'ASSET {name!r}: {ps!r} -> {pd!r}')
    return quot + pd + quot


def path_to_uid(path):
    # .../app/gws/plugin/qgis/_demo/flat.cx -> gws.plugin.qgis._demo.flat
    return (
        path
        .split('.')[0]
        .replace(APP_DIR, '')
        .strip('/')
        .replace('/', '.')
    )


if __name__ == '__main__':
    cli.main('demo.py', main, USAGE)

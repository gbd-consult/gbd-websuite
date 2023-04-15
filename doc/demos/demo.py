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
    -appdir
        'app' directory root

    -include <path>
        include a file in the config

    -only <pattern>
        a regex pattern to use only specific demos
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

    # relocate assets

    assets = {
        os.path.basename(path): path
        for path in all_paths
        if not path.endswith('.cx')
    }

    appdir = args.get('appdir') or APP_DIR

    def relocate_path(val):
        if isinstance(val, list):
            return [relocate_path(v) for v in val]
        if isinstance(val, dict):
            return {k: relocate_path(v) for k, v in val.items()}
        if isinstance(val, str) and val in assets:
            r = assets[val].replace(APP_DIR, appdir)
            # print(val, '->', r)
            return r
        return val

    config = relocate_path(config)

    # all done!

    print(json.dumps(config, indent=4, ensure_ascii=False))


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


def read_config_file(path):
    text = cli.read_file(path)
    if START_PROJECT in text:
        # if this is a project config, inject an uid from its path
        text = text.lstrip()
        uid = path_to_uid(path)
        text = text.replace(
            START_PROJECT,
            START_PROJECT + f' uid "{uid}"\n'
        )
    return text + '\n\n'


def path_to_uid(path):
    # .../app/gws/plugin/qgis/_demo/flat.cx -> gws.plugin.qgis._demo.flat
    p = path.split('.')[0].split('/')
    if 'gws' in p:
        p = p[p.index('gws'):]
    return '.'.join(p)


if __name__ == '__main__':
    cli.main('demo.py', main, USAGE)

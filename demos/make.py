"""Combine demos from the source tree into a single config."""

import os
import sys
import re
import json

import mistune

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../app')
sys.path.insert(0, APP_DIR)

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
    # basic config - from this dir

    main_cfg = [
        read_config_file(THIS_DIR + '/config_base.cx')
    ]
    projects_cfg = []

    only = args.get('only')
    cx_paths = list(cli.find_files(APP_DIR, r'_demo/.+?\.cx$'))

    # include configs from app subdirectories

    for path in cx_paths:
        if path.endswith('/_app.cx'):
            # application config
            main_cfg.append(read_config_file(path))
            continue

        if not only or re.search(only, path):
            # project configuration
            cfg = read_config_file(path)
            projects_cfg.append(f'''
                projects+ {{ 
                    {cfg}
                    metadata.authorityIdentifier "app{path.replace(APP_DIR, '')}"
                }}
            ''')

    config = '\n'.join(main_cfg) + '\n' + '\n'.join(projects_cfg)

    # extra include

    if args.get('include'):
        config += '\n@include ' + args.get('include') + '\n'

    # parse config

    try:
        config = gws.lib.vendor.jump.render(config)
    except gws.lib.vendor.jump.CompileError as exc:
        error_lines(config, exc.line)
        return cli.fatal(str(exc))

    try:
        config_dct = gws.lib.vendor.slon.parse(config, as_object=True)
    except gws.lib.vendor.slon.SlonError as exc:
        error_lines(config, exc.args[2])
        return cli.fatal(str(exc))

    projects = config_dct.get('projects')
    if not projects:
        return cli.fatal('no demo projects found')

    # render abstracts

    markdown = mistune.create_markdown()
    for prj in projects:
        text = prj.get('metadata', {}).get('abstract', '')
        if text:
            text = markdown(text)

            # some annoyances
            text = re.sub(r'\s+(</code>)', r'\1', text)
            text = re.sub(r'<a', r'<a target="_blank"', text)

            prj['metadata']['abstract'] = text

    # check uids

    uids = set()
    for prj in projects:
        uid = prj.get('uid')
        if not uid:
            cli.fatal(f'no uid for project {prj!r}')
        if uid in uids:
            cli.fatal(f'duplicate uid {uid!r} for project {prj!r}')
        uids.add(uid)

    # all done!

    res = json.dumps(config_dct, indent=4, ensure_ascii=False)
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
    text = relocate_assets(text, path)
    return text


def relocate_assets(text, path):
    # replace asset paths with absolute paths
    return re.sub(RE_ASSET, lambda m: relocate_asset(m, path), text)


def relocate_asset(m, path):
    quot, name = m.groups()
    ps = os.path.realpath(os.path.join(os.path.dirname(path), name))
    if not ps.startswith(APP_DIR):
        return m.group(0)
    pd = ps.replace(APP_DIR, '/gws-app')
    return quot + pd + quot


if __name__ == '__main__':
    cli.main('demo.py', main, USAGE)

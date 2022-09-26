"""Combine demos from the source tree into a single config."""

import os
import sys
import re
import json

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.abspath(os.path.dirname(__file__) + '/../../app')
sys.path.append(APP_DIR)

from gws.lib.vendor import jump, slon


def main():
    extra_include = sys.argv[1] if len(sys.argv) > 1 else None
    include_only = sys.argv[2] if len(sys.argv) > 2 else None

    tpl = [
        f'@include {path}'
        for path in sorted(find_files(CUR_DIR, r'\.cx$'))
    ]

    if extra_include:
        tpl.append(f'@include {extra_include}')

    tpl.append('@demo_init')

    for path in find_files(APP_DIR, r'demos/.+?\.cx$'):
        if not include_only or include_only in path:
            tpl.append(f'@include {path}')

    tpl = '\n'.join(tpl)
    src = jump.render(tpl, loader=inject_project_uid)
    src = src.strip()

    try:
        cfg = slon.parse(src, as_object=True)
    except slon.SlonError as exc:
        lpr(src, exc.args[2])
        raise

    print(json.dumps(cfg, indent=4))


def lpr(src, mark=None):
    lines = src.split('\n')
    maxl = len(str(len(lines)))
    for n, s in enumerate(lines, 1):
        if n == mark:
            print('>>>', end=' ')
        print(f'{n:{maxl}d}:  {s}')


def path_to_uid(path):
    p = path.split('.')[0].split('/')
    while p[0] != 'gws':
        p.pop(0)
    return '.'.join(p)


START_PROJECT = 'projects+ {'


def inject_project_uid(_, path):
    with open(path) as fp:
        text = fp.read()
    if text.lstrip().startswith(START_PROJECT):
        text = text.lstrip()
        uid = path_to_uid(path)
        text = START_PROJECT + f' uid "{uid}"\n' + text[len(START_PROJECT):]
    return text, path


def find_files(dirname, pattern=None, deep=True):
    if not os.path.isdir(dirname):
        return

    de: os.DirEntry
    for de in os.scandir(dirname):
        if de.name.startswith('.'):
            continue
        if de.is_dir() and deep:
            yield from find_files(de.path, pattern)
            continue
        if de.is_file() and (pattern is None or re.search(pattern, de.path)):
            yield de.path


if __name__ == '__main__':
    main()

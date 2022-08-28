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
    def lpr(src):
        for n, s in enumerate(src.split('\n'), 1):
            print(n, s)

    tpl = []
    tpl.extend(f'@include {path}' for path in sorted(find_files(CUR_DIR, r'\.cx$')))
    if len(sys.argv) > 1:
        tpl.append(f'@include {sys.argv[1]}')
    tpl.append('@demo_init')
    tpl.extend(f'@include {path}' for path in sorted(find_files(APP_DIR, r'\.cx$')))

    tpl = '\n'.join(tpl)

    src = jump.render(tpl)

    cfg = slon.parse(src, as_object=True)

    print(json.dumps(cfg, indent=4))


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

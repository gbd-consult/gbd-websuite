import sys
import os
import re


def inject_version(version, path):
    with open(path) as fp:
        t = fp.read()
    t = re.sub(r'VERSION\s*=\s*.+', 'VERSION=%r' % version, t)
    with open(path, 'w') as fp:
        fp.write(t)


if __name__ == '__main__':

    cdir = os.path.dirname(__file__)
    base = os.path.abspath(cdir + '/..')

    with open(cdir + '/../VERSION') as fp:
        VERSION = fp.read().strip()

    paths = [
        base + '/app/gws/core/const.py',
        base + '/client/options.js',
        base + '/doc/sphinx/conf.py',
    ]

    for path in paths:
        inject_version(VERSION, path)

    source_dir = os.path.abspath(base + '/app/gws')
    out_dir = os.path.abspath(base + '/app/spec/gen')
    os.makedirs(out_dir, exist_ok=True)

    sys.path.append(cdir)
    # noinspection PyUnresolvedReferences
    import impl.main

    impl.main.run(source_dir, out_dir, VERSION)

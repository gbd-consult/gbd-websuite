import sys
import os
import re


def get_version(path):
    with open(path) as fp:
        for s in fp:
            m = re.search(r"VERSION\s*=\s*'(.+?)'", s)
            if m:
                return m.group(1)


if __name__ == '__main__':

    cdir = os.path.dirname(__file__)
    VERSION = get_version(cdir + '/../app/gws/core/const.py')

    try:
        source_dir = sys.argv[1]
        out_dir = sys.argv[2]
    except IndexError:
        source_dir = os.path.abspath(cdir + '/../app/gws')
        out_dir = os.path.abspath(cdir + '/../app/spec/en')

    sys.path.append(cdir)
    # noinspection PyUnresolvedReferences
    import impl.main

    impl.main.run(source_dir, out_dir, VERSION)

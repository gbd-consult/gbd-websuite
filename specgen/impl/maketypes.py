"""Create types/init.py from from the .in.py template"""

import re

from . import makestubs

def run(source_dir, stubs):
    d = source_dir + '/types'
    src_path = d + '/__init__.in.py'
    dst_path = d + '/__init__.py'

    with open(src_path) as fp:
        src = fp.read()

    dst = re.sub(r'#\s*@include(.+)', lambda m: _include(d, m.group(1)), src)

    dst += makestubs.run(stubs)

    with open(dst_path, 'w') as fp:
        fp.write(dst)


def _include(d, path):
    path = d + '/' + path.strip()
    with open(path + '.py') as fp:
        src = fp.read()
    src = re.sub(r'from(.+?)import(.+)', '', src)
    return src

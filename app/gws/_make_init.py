import os
import re

GWS_DIR = os.path.dirname(__file__)


def make_ext():
    commands = [
        'api',
        'get',
        'post',
        'cli',
    ]

    types_file = f'{GWS_DIR}/ext/types.txt'
    types = [s.strip() for s in read_file(types_file).strip().split('\n')]

    src = f'{GWS_DIR}/ext/__init__.py'
    dst = f'{GWS_DIR}/ext/__init__.py'

    text = read_file(src)
    text = text.split('##')[0].strip() + '\n\n##'

    text += f"\n\n\nTYPES = [\n"
    for t in types:
        text += f'    "{t}",\n'
    text += ']\n'

    text += f"\n\nclass command:\n"
    for t in sorted(commands):
        text += f"    class {t}(_methodTag): pass\n"

    text += f"\n\nclass _new:\n"
    for t in sorted(types):
        text += f"    def {t}(self, *args): pass\n"
    text += f"\n\nnew = _new()\n"

    for k in 'object', 'config', 'props':
        text += f"\n\nclass {k}:\n"
        for t in sorted(types):
            text += f"    class {t} (_classTag): extName = 'gws.ext.{k}.{t}'\n"

    write_file(dst, text)


def make_gws_init():
    src = f'{GWS_DIR}/__init__.pyinc'
    dst = f'{GWS_DIR}/__init__.py'
    text = process_includes(src, [])
    write_file(dst, text)


def process_includes(path, stack):
    if path in stack:
        raise ValueError(f'circular include {path=}')

    text = read_file(path)
    lines = []

    sep = '#' * 80

    if stack:
        lines.append('')
        lines.append(sep)
        lines.append(f'# {path[len(GWS_DIR):]}')
        lines.append('')
        lines.append('')

    for ln in text.strip().split('\n'):
        m = re.match(r'# @include (\S+)', ln)
        if not m:
            lines.append(ln)
            continue
        inc = os.path.dirname(path) + '/' + m.group(1)
        lines.append(process_includes(inc, stack + [path]))

    if stack:
        lines.append(sep)
        lines.append('')

    return '\n'.join(lines)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file(path, text):
    try:
        old = read_file(path)
    except:
        old = ''
    if text == old:
        return
    print(f'[make_init] updated {path}')
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


if __name__ == '__main__':
    make_ext()
    make_gws_init()

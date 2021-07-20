"""Spec generator CLI tool

This tool is supposed to be invoked on the _host_ (developer) system
to generate developer specs (python stubs, typescript interfaces etc)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

__package__ = 'generator'

from . import base, main


def parse_args(argv):
    args = []
    kwargs = {}
    key = None

    for a in argv:
        if a.startswith('--'):
            key = a[2:]
            kwargs[key] = True
        elif a.startswith('-'):
            key = a[1:]
            kwargs[key] = True
        elif key:
            kwargs[key] = a
            key = None
        else:
            args.append(a)

    return args, base.Data(**kwargs)


_COMMANDS = ('build', 'server')

if __name__ == '__main__':
    args, opts = parse_args(sys.argv[1:])

    cmd = args[0] if args else _COMMANDS[0]
    if cmd not in _COMMANDS:
        print(f'invalid command, expected {_COMMANDS!r}')
        sys.exit(1)

    base.log.set_level('DEBUG' if opts.v or opts.verbose else (opts.loglevel or 'INFO'))

    try:
        if cmd == 'build':
            main.generate_for_build(opts.manifest)
        if cmd == 'server':
            main.generate_for_server(opts.manifest)
    except base.Error as e:
        print('-' * 40)
        print('SPEC GENERATOR ERROR:', e.args[0])
        print('-' * 40)
        raise
    except Exception as e:
        print('-' * 40)
        print('UNHANDLED SPEC GENERATOR ERROR:', repr(e))
        print('-' * 40)
        raise

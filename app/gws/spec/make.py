"""Spec generator CLI tool

This tool is supposed to be invoked on the _host_ (developer) system
to generate developer specs (python stubs, typescript interfaces etc)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from generator import generator


def main():
    args = parse_args(sys.argv)
    if not args.get('out'):
        raise ValueError('output directory required')

    try:
        generator.generate_and_store(
            out_dir=args.get('out'),
            root_dir=args.get('root'),
            manifest_path=args.get('manifest'),
            debug=args.get('v'))

    except generator.Error as e:
        print('-' * 40)
        print('SPEC GENERATOR ERROR:', e.args[0])
        print('-' * 40)
        raise
    except Exception as e:
        print('-' * 40)
        print('UNHANDLED SPEC GENERATOR ERROR:', repr(e))
        print('-' * 40)
        raise


def parse_args(argv):
    args = {}
    opt = None
    n = 0

    for a in argv:
        if a.startswith('--'):
            opt = a[2:]
            args[opt] = True
        elif a.startswith('-'):
            opt = a[1:]
            args[opt] = True
        elif opt:
            args[opt] = a
            opt = None
        else:
            args[n] = a
            n += 1

    return args


if __name__ == '__main__':
    main()

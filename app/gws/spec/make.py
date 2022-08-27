"""Spec generator CLI tool

This tool is supposed to be invoked on the _host_ (developer) system
to generate developer specs (python stubs, typescript interfaces etc)
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../..'))

from generator import base, main, util

if __name__ == '__main__':
    args = util.parse_args(sys.argv)
    if not args.get('out'):
        raise ValueError('output directory required')

    try:
        main.generate_and_store(
            out_dir=args.get('out'),
            root_dir=args.get('root'),
            manifest_path=args.get('manifest'),
            debug=args.get('v'))

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

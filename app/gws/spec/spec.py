"""Spec generator CLI tool

This tool is supposed to be invoked on the _host_ (developer) system
to generate developer specs (python stubs, typescript interfaces etc)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../../../app'))

import gws.lib.cli as cli

import gws.spec.generator.generator as generator

USAGE = """
GWS Spec Compiler
~~~~~~~~~~~~~~~~~
  
    python3 spec.py <output-directory> <options>

Options:

    -root <dir>
        application root directory ('app')

    -manifest <path>
        path to MANIFEST.json

    -v
        verbose logging
"""


def main(args):
    out_dir = args.get(1)
    if not out_dir:
        cli.fatal('output directory required')

    os.makedirs(out_dir, exist_ok=True)

    try:
        generator.generate_and_write(
            out_dir=out_dir,
            root_dir=args.get('root'),
            manifest_path=args.get('manifest'),
            debug=args.get('v'))

    except generator.Error as e:
        cli.error('-' * 40)
        cli.error(f'SPEC GENERATOR ERROR: {e.args[0]}')
        cli.error('-' * 40)
        raise
    except Exception as e:
        cli.error('-' * 40)
        cli.error(f'UNHANDLED SPEC GENERATOR ERROR: {e!r}')
        cli.error('-' * 40)
        raise


if __name__ == '__main__':
    cli.main('spec', main, USAGE)

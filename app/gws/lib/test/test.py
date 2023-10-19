"""Test configurator and invoker.

This file provides utilities for the test runner
on the host machine (see `make test` and `app/test.py`).
The purpose of these utils is to create a docker compose file, start the compose
and invoke the test runner inside the GWS container (`gws test`).
"""

import os
import sys

import pytest

APP_DIR = os.path.realpath(os.path.dirname(__file__) + '/../../../')
sys.path.insert(0, APP_DIR)

import gws
import gws.spec.runtime
import gws.lib.cli as cli
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.test.util as u
import gws.lib.test.configure as configure

USAGE = """
GWS test runner
~~~~~~~~~~~~~~~

    python3 test.py <command> <options>

Commands:

    configure  - configure the test environment
    go         - start the test environment, run tests and stop
    run        - run tests
    restart    - restart the test environment
    start      - start the test environment
    stop       - stop the test environment
        
Options:

    --only <regex>        - only run filenames matching the pattern 
    --manifest <manifest> - path to MANIFEST.json
    --verbose             - enable debug logging
    
    Additionally, you can pass any pytest option:
    https://docs.pytest.org/latest/reference.html#command-line-flags

"""


def main(args):
    cmd = args.get(1)

    if cmd == 'configure':
        manifest_text = ''
        s = args.get('manifest')
        if s:
            manifest_text = gws.read_file(s)
        do_configure(manifest_text)
        cli.info(f'tests configured in {u.work_dir()!r}')
        return 0

    if cmd == 'run':
        do_run(args)
        return 0

    cli.fatal('invalid arguments, try test.py -h for help')


##

def do_configure(manifest_text):
    wd = u.work_dir()
    configure.empty_dir(wd)
    gws.write_file(f'{wd}/MANIFEST.json', manifest_text or '{}')
    gws.write_file(f'{wd}/docker-compose.yml', configure.compose_yml())
    gws.write_file(f'{wd}/pg_service.conf', configure.pg_service_conf())
    gws.write_file(f'{wd}/pytest.ini', configure.pytest_ini())


def do_run(args):
    gws.ensure_system_dirs()

    files = u.enumerate_files_for_test(args.get('only'))
    if not files:
        gws.log.error(f'no files to test')
        return

    # verbosity

    gws.log.set_level('INFO')
    verbose = args.get('verbose') or args.get('v')
    if verbose:
        gws.log.set_level('DEBUG')

    # run pytest

    pytest_args = ['-c', f'{u.work_dir()}/pytest.ini', '--rootdir', f'{u.APP_DIR}/gws']
    rest = args.get('_rest')
    if rest:
        pytest_args.extend(rest)
    pytest_args.extend(files)
    gws.log.info(f'running pytest with args: {pytest_args}')

    try:
        pytest.main(pytest_args)
    except:
        gws.log.exception()


##

if __name__ == '__main__':
    cli.main('test', main, USAGE)

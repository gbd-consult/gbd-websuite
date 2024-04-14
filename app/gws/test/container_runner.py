"""Test runner (container).

Container test runner. Assumes tests are configured on the host with ``host_runner configure``.

"""

import time
import re
import urllib.request

import psycopg2
import pytest

import gws
import gws.lib.cli as cli
import gws.lib.jsonx
import gws.lib.net
import gws.lib.osx
import gws.spec.runtime
import gws.test.util as u

USAGE = """
GWS in-container test runner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    python3 runner.py <options> - <pytest options>

Options:

    --work_dir <path>     - path to the local 'ini' file 
    --only <regex>        - only run filenames matching the pattern 
    --verbose             - enable debug logging
    
Pytest options:
    see https://docs.pytest.org/latest/reference.html#command-line-flags

"""


def main(args):
    gws.u.ensure_system_dirs()

    wd = args.get('work_dir')
    u.OPTIONS = gws.lib.jsonx.from_path(f'{wd}/options.json')
    u.OPTIONS['runner.work_dir'] = wd

    pytest_args = [
        f'--config-file={wd}/pytest.ini',
        f'--rootdir=/gws-app',
        f'--ignore-glob=__build',
    ]
    pytest_args.extend(args.get('_rest', []))

    gws.log.set_level('INFO')
    if args.get('verbose') or args.get('v'):
        gws.log.set_level('DEBUG')
        pytest_args.append('--tb=native')
        pytest_args.append('--verbose')

    files = enum_files_for_test(args.get('only') or args.get('o'))
    if not files:
        cli.fatal(f'no files to test')
        return
    pytest_args.extend(files)

    if not health_check():
        cli.fatal('health check failed')
        return

    cli.info('pytest ' + ' '.join(pytest_args))
    pytest.main(pytest_args)


##


def enum_files_for_test(only_pattern):
    """Enumerate files to test, wrt --only option."""

    regex = u.OPTIONS.get('pytest.python_files').replace('*', '.*')

    files = list(gws.lib.osx.find_files(f'{gws.c.APP_DIR}/gws', regex))
    if only_pattern:
        files = [f for f in files if re.search(only_pattern, f)]

    # sort files semantically

    _sort_order = ['/core/', '/lib/', '/base/', '/plugin/']

    def _sort_key(path):
        for n, s in enumerate(_sort_order):
            if s in path:
                return n, path
        return 99, path

    files.sort(key=_sort_key)
    return files


##

_HEALTH_CHECK_ATTEMPTS = 10
_HEALTH_CHECK_PAUSE = 3


def health_check():
    status = {s: False for s in u.OPTIONS['runner.services']}

    for _ in range(_HEALTH_CHECK_ATTEMPTS):
        for s, ok in status.items():
            if not ok:
                fn = globals().get(f'health_check_service_{s}')
                err = fn() if fn else None
                if err:
                    cli.info(f'waiting for {s!r}: {err}')
                status[s] = not err
        if all(status.values()):
            return True
        time.sleep(_HEALTH_CHECK_PAUSE)

    return False


def health_check_service_postgres():
    try:
        conn = psycopg2.connect(
            host=u.OPTIONS['service.postgres.host'],
            port=u.OPTIONS['service.postgres.port'],
            user=u.OPTIONS['service.postgres.user'],
            password=u.OPTIONS['service.postgres.password'],
            database=u.OPTIONS['service.postgres.database'],
            connect_timeout=1
        )
    except psycopg2.Error as exc:
        return repr(exc)
    conn.close()


def health_check_service_qgis():
    return http_ping(u.OPTIONS['service.qgis.host'], u.OPTIONS['service.qgis.port'])


def health_check_service_mockserver():
    return http_ping(u.OPTIONS['service.mockserver.host'], u.OPTIONS['service.mockserver.port'])


def http_ping(host, port):
    url = f'http://{host}:{port}'
    try:
        with urllib.request.urlopen(url) as res:
            if res.status == 200:
                return
            return f'http status={res.status}'
    except Exception as exc:
        return repr(exc)


##


if __name__ == '__main__':
    cli.main('test', main, USAGE)

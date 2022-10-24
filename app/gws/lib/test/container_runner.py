"""Test runner.

This is supposed to be run in the container.
Accepts the same arguments as the host runner:
(`--manifest <manifest> --only <patterns> pytest-opts`)
"""

import configparser
import re
import time

import pytest
import psycopg2

import gws
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.net
import gws.spec.runtime

from . import glob

TEST_FILE_RE = r'_test\.py$'
TEST_FILE_GLOB = '*_test.py'

HEALTH_CHECK_PAUSE = 5


def main(argv):
    argv = argv[1:]

    gws.ensure_system_dirs()

    # read the test config, created by the host runner

    cfg_path = gws.lib.osx.getenv('GWS_TEST_CONFIG')
    if cfg_path:
        glob.CONFIG.update(gws.lib.jsonx.from_path(cfg_path))

    wd = glob.CONFIG['runner.work_dir']

    # load the passed manifest or the default one

    manifest_path = poparg2(argv, '--manifest')
    if not manifest_path:
        manifest_path = wd + '/MANIFEST.json'
        gws.lib.jsonx.to_path(manifest_path, glob.DEFAULT_MANIFEST)
    gws.spec.runtime.create(manifest_path, read_cache=True, write_cache=True)
    glob.CONFIG['manifest_path'] = manifest_path

    # enumerate files to test, wrt --only

    rootdir = gws.APP_DIR + '/gws'
    files = list(gws.lib.osx.find_files(rootdir, TEST_FILE_RE))

    only = poparg2(argv, '--only')
    if only:
        files = [f for f in files if re.search(only, f)]

    if not files:
        gws.log.error(f'no files to test')
        return

    _sort_order = ['/core/', '/lib/', '/base/', '/plugin/']

    def _sort_key(path):
        for n, s in enumerate(_sort_order):
            if s in path:
                return n, path
        return 99, path

    files.sort(key=_sort_key)

    # verbosity

    gws.log.set_level('INFO')
    verbose = poparg1(argv, '--verbose') or poparg1(argv, '-v')
    if verbose:
        gws.log.set_level('DEBUG')

    # create pytest.ini

    cc = configparser.ConfigParser()
    cc.add_section('pytest')
    for k, v in glob.CONFIG.items():
        k = k.split('.')
        if k[0] == 'pytest':
            cc.set('pytest', k[1], v)
    cc.set('pytest', 'python_files', TEST_FILE_GLOB)
    cc.set('pytest', 'cache_dir', f'{wd}/pytest_cache')

    pytest_ini_path = wd + '/pytest.ini'
    with open(pytest_ini_path, 'wt') as fp:
        cc.write(fp)

    # check services

    if not health_check():
        gws.log.error(f'health check failed, exiting')
        return 255

    # run pytest

    pytest_args = ['-c', pytest_ini_path, '--rootdir', rootdir]
    pytest_args.extend(argv)
    pytest_args.extend(files)
    gws.log.info(f'running pytest with args: {pytest_args}')
    pytest.main(pytest_args)


# service health checks

def health_check():
    ok = {s: False for s in glob.CONFIG['runner.services'].split()}

    for _ in range(int(glob.CONFIG['runner.health_check_attempts'])):
        for s in ok:
            if not ok[s]:
                fn = globals()[f'health_check_for_service_{s}']
                ok[s] = fn()
                gws.log.info(f'service {s}: ' + ('OK' if ok[s] else 'waiting...'))
        if all(ok.values()):
            return True
        time.sleep(HEALTH_CHECK_PAUSE)

    return False


def health_check_for_service_gws():
    return True


def health_check_for_service_postgres():
    try:
        conn = psycopg2.connect(
            host=glob.CONFIG['runner.host_name'],
            port=glob.CONFIG['service.postgres.port'],
            user=glob.CONFIG['service.postgres.user'],
            password=glob.CONFIG['service.postgres.password'],
            database=glob.CONFIG['service.postgres.database'],
            connect_timeout=1
        )
    except psycopg2.Error:
        return False
    conn.close()
    return True


def health_check_for_service_qgis():
    return http_ok(glob.CONFIG['runner.host_name'] + ':' + glob.CONFIG['service.qgis.port'])


def health_check_for_service_mockserv():
    return http_ok(glob.CONFIG['runner.host_name'] + ':' + glob.CONFIG['service.mockserv.port'])


# utils

def http_ok(url):
    url = 'http://' + url
    try:
        res = gws.lib.net.http_request(url, timeout=1)
    except gws.lib.net.Error:
        return False
    return res and res.ok


def poparg1(argv, key):
    try:
        n = argv.index(key)
    except ValueError:
        return
    argv.pop(n)
    return True


def poparg2(argv, key):
    try:
        n = argv.index(key)
    except ValueError:
        return
    argv.pop(n)
    try:
        return argv.pop(n)
    except IndexError:
        return

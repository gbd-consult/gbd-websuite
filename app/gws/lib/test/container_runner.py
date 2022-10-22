"""Test runner.

This is supposed to be run in the container.
Accepts the same arguments as the host runner:
(`--manifest <manifest> --only <patterns> pytest-opts`)
"""

import configparser
import re

import pytest

import gws
import gws.lib.jsonx
import gws.lib.osx
import gws.spec.runtime
from . import glob

TEST_FILE_RE = r'_test\.py$'
TEST_FILE_GLOB = '*_test.py'


def main(argv):
    argv = argv[1:]

    gws.ensure_system_dirs()

    # read the test config, created by the host runner

    cfg_path = gws.lib.osx.getenv('GWS_TEST_CONFIG')
    if cfg_path:
        glob.CONFIG.update(gws.lib.jsonx.from_path(cfg_path))

    wd = glob.CONFIG['runner.work_dir']

    # load the passed manifest or the default one

    manifest_path = poparg(argv, '--manifest')
    if not manifest_path:
        manifest_path = wd + '/DEFAULT_MANIFEST.json'
        gws.lib.jsonx.to_path(manifest_path, glob.DEFAULT_MANIFEST)
    gws.spec.runtime.create(manifest_path, read_cache=True, write_cache=True)
    glob.CONFIG['manifest_path'] = manifest_path

    # enumerate files to test, wrt --only

    rootdir = gws.APP_DIR + '/gws'
    files = list(gws.lib.osx.find_files(rootdir, TEST_FILE_RE))

    only = poparg(argv, '--only')
    if only:
        only_patterns = [r.strip() for r in only.split(',')]
        files = [f for f in files if any(re.search(r, f) for r in only_patterns)]

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

    # run pytest

    pytest_args = ['-c', pytest_ini_path, '--rootdir', rootdir]
    pytest_args.extend(argv)
    pytest_args.extend(files)
    gws.log.info(f'running pytest with args: {pytest_args}')
    pytest.main(pytest_args)


def poparg(argv, key):
    try:
        n = argv.index(key)
    except ValueError:
        return
    argv.pop(n)
    try:
        return argv.pop(n)
    except IndexError:
        return

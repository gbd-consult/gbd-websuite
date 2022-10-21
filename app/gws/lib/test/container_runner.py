"""Test runner."""

import sys
import pytest

import gws
import gws.spec.runtime
import gws.lib.jsonx
import gws.lib.osx

from . import glob


def main():
    args = sys.argv[1:]

    gws.ensure_system_dirs()

    glob.CONFIG.update(gws.lib.jsonx.from_path(glob.TEST_CONFIG_PATH_IN_CONTAINER))

    manifest = glob.CONFIG.get('manifest')
    gws.lib.jsonx.to_path(glob.MANIFEST_PATH, manifest or glob.DEFAULT_MANIFEST)

    rootdir = gws.APP_DIR + '/gws'
    files = list(gws.lib.osx.find_files(rootdir, r'_test\.py'))
    spec = True

    if args and not args[0].startswith('-'):
        pattern = args.pop(0)
        if pattern.startswith('nospec:'):
            pattern = pattern.split(':')[1]
            spec = False
        if pattern:
            files = [f for f in files if pattern in f]

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

    if spec:
        gws.spec.runtime.create(glob.MANIFEST_PATH, read_cache=True, write_cache=True)

    pytest_args = ['-c', glob.CONFIG['pytest_ini_path'], '--rootdir', rootdir]
    pytest_args.extend(args)
    pytest_args.extend(files)
    gws.log.debug(f'running pytest with args: {pytest_args}')
    pytest.main(pytest_args)

"""Misc test utilities."""

import inspect
import os
import shutil
import time

import pytest

import gws
import gws.base.web.web_app
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.password

from . import glob, mockserv


def setup():
    gws.log.debug(f'TEST:setup')
    pass


def teardown():
    gws.log.debug(f'TEST:teardown')

    gws.lib.osx.unlink(glob.SESSION_STORE_PATH)
    gws.base.web.web_app.reload()
    gws.config.deactivate()
    mockserv.command('reset')


def make_users_json(lst):
    path = '/tmp/gws_test_users.json'
    if lst is None:
        gws.lib.osx.unlink(path)
        return None

    for v in lst:
        v['password'] = gws.lib.password.encode(v['password'])
    gws.lib.jsonx.to_path(path, lst)
    return path


def write_file(path, text):
    pp = gws.lib.osx.parse_path(path)
    if pp['dirname'].startswith(glob.TEMP_DIR):
        gws.ensure_dir(pp['dirname'])
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read()


def write_file_if_changed(path, text):
    curr = read_file(path)
    if text != curr:
        write_file(path, text)


def copy_file(path, dir):
    shutil.copy(path, dir)


def rel_path(path):
    f = inspect.stack(2)[1].filename
    return os.path.join(os.path.dirname(f), path)


def sleep(n):
    time.sleep(n)


def raises(exc):
    return pytest.raises(exc)


def dict_of(x):
    if gws.is_data_object(x):
        # noinspection PyTypeChecker
        return dict(sorted(vars(x).items()))
    return x

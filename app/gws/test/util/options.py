"""Test runner options."""

import os

import gws
import gws.lib.jsonx
import gws.lib.osx

_data = {}


def option(name, default=None):
    return _data.get(name, default)


def load_options(base_dir):
    _data.clear()
    _data.update(gws.lib.jsonx.from_path(f'{base_dir}/config/OPTIONS.json'))
    _data['BASE_DIR'] = base_dir

    tmp_dir = f'{base_dir}/tmp'
    if os.path.isdir(tmp_dir):
        gws.lib.osx.rmdir(tmp_dir)
    gws.u.ensure_dir(tmp_dir)

    if not gws.env.GWS_IN_CONTAINER:
        # if we are not in a container, use 'localhost:exposed_port' for all services
        for k, v in list(_data.items()):
            if k.endswith('.host'):
                _data[k] = 'localhost'
            if k.endswith('.port'):
                _data[k] = _data[k.replace('.port', '.expose_port')]

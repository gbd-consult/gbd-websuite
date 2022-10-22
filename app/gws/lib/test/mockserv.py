"""Utilities to interact with the mock server."""

import gws
import gws.lib.net
import gws.lib.jsonx

from .glob import CONFIG


def command(cmd, params=None):
    params = params or {}
    params['cmd'] = cmd
    res = gws.lib.net.http_request(
        url(''),
        data=gws.lib.jsonx.to_string(params),
        method='post'
    )
    return gws.lib.jsonx.from_string(res.text)


def poke(pattern, response):
    return command('poke', {'pattern': pattern, 'response': response})


def begin_capture():
    return command('begin_capture')


def end_capture():
    res = command('end_capture')
    return [gws.lib.net.parse_url('http://host' + u) for u in res['urls']]


def create_wms(config):
    command('create_wms', {'config': config})


def url(u):
    base_url = f"http://{CONFIG['runner.host_name']}:{CONFIG['service.mockserv.port']}"
    return base_url + '/' + u

import time

import gws
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.vendor.slon
import gws.lib.net
import gws.server.control

from . import glob


def root():
    return gws.config.root()


def configure(config, parse=True):
    def _dct2cfg(d):
        if isinstance(d, dict):
            return gws.Config({k: _dct2cfg(v) for k, v in d.items()})
        if isinstance(d, (list, tuple)):
            return [_dct2cfg(v) for v in d]
        return d

    gws.log.debug(f'TEST:configure')

    config_defaults = {
        'server': {
            'log': {
                'level': 'DEBUG'
            },
            'mapproxy': {
                'enabled': True,
                'workers': 1,
                'forceStart': True,
            },
            'monitor': {
                'enabled': False
            },
            'qgis': {
                'host': glob.CONFIG['runner.host_name'],
                'port': glob.CONFIG['service.qgis.port'],
            },
            'web': {
                'enabled': True,
                'workers': 1
            },
            'spool': {
                'enabled': True,
                'workers': 1
            },
            'timeout': 60
        },
        'auth': {
            'sessionStore': 'sqlite',
            'sessionStorePath': glob.SESSION_STORE_PATH,
        },
    }

    if isinstance(config, str):
        config = gws.lib.vendor.slon.parse(config, as_object=True)
    dct = gws.deep_merge(config_defaults, config)

    config = _dct2cfg(dct)
    gws.lib.jsonx.to_path(glob.GWS_CONFIG_PATH, config, pretty=True)

    if parse:
        r = gws.config.configure(manifest_path=glob.MANIFEST_PATH, config_path=glob.GWS_CONFIG_PATH)
    else:
        r = gws.config.configure(manifest_path=glob.MANIFEST_PATH, config=config)

    gws.config.activate(r)
    gws.config.store(r)

    return r


def configure_and_reload(config, parse=True):
    r = configure(config, parse)
    gws.server.control.reload(['mapproxy', 'web'])

    for service in 'http', 'mpx':
        _wait_for_port(service)

    return r


def _wait_for_port(service):
    while 1:
        port = glob.CONFIG[f'service.gws.{service}_port']
        url = 'http://' + glob.CONFIG['runner.host_name'] + ':' + str(port)
        res = gws.lib.net.http_request(url)
        if res.ok:
            return
        gws.log.debug(f'TEST:waiting for {service}:{port}')
        time.sleep(2)

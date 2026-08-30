"""Mockmap HTTP server: routing, configuration, delay."""

import http.server
import os
import signal
import sys
import threading
import time
import urllib.parse
from typing import cast

import gws
import gws.lib.vendor.jump
import gws.lib.vendor.slon

from . import scene, services

HOST = '0.0.0.0'
PORT = 80
DELAY = int(os.environ.get('MOCKMAP_DELAY') or 0)

_lock = threading.Lock()
_config_path = os.path.dirname(__file__) + '/config.cx'
_config_repr = ''
_services: dict[str, services.Service] = {}
_scenes: dict[str, scene.Scene] = {}


def load_config() -> list[dict]:
    text = gws.lib.vendor.jump.render_path(
        _config_path,
        {
            'true': True,
            'false': False,
            'env': dict(os.environ),
            'gws': gws,
        },
    )
    cfg = cast(dict, gws.lib.vendor.slon.loads(text, as_object=True))
    dfl = cfg.get('default') or {}
    return [{**dfl, **s} for s in cfg['services']]


def reload_config():
    global _config_repr

    cfgs = load_config()
    rep = repr(cfgs)

    with _lock:
        if rep == _config_repr:
            return
        _services.clear()
        _scenes.clear()
        services._cache.clear()
        for cfg in cfgs:
            scn = scene.Scene(cfg)
            scn = _scenes.setdefault(scn.key, scn)
            _services[cfg['uid']] = services.create(cfg, scn)
        _config_repr = rep
        _writeln(f'[mockmap] loaded {len(_services)} services: {" ".join(sorted(_services))}')


class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        try:
            self.run()
        except Exception as exc:
            for ln in gws.log.exception_backtrace(exc):
                _writeln(f'[mockmap] ERROR {ln}')
            self.end('Internal Server Error', 500, 'text/plain')

    def log_message(self, fmt, *args):
        pass

    def run(self):
        reload_config()
        _writeln(f'[mockmap] {self.command} {self.path}')

        path, _, qs = self.path.partition('?')
        parts = [p for p in path.split('/') if p]

        if not parts:
            return self.end('\n'.join(sorted(_services)) + '\n', 404, 'text/plain')

        uid, rest = parts[0], parts[1:]
        srv = _services.get(uid)
        if not srv:
            return self.end(f'no such service: {uid}', 404, 'text/plain')

        query = {k.lower(): v[0] for k, v in urllib.parse.parse_qs(qs).items()}
        base = f'http://{self.headers.get("host")}/{uid}'

        kind, res = srv.handle(rest, query, base)

        if DELAY > 0:
            time.sleep(DELAY / 1000.0)

        if kind == 'image':
            return self.end(res, 200, 'image/png')
        return self.end(res, 200, 'text/xml; charset=utf-8')

    def end(self, content, status, content_type):
        body = content if isinstance(content, bytes) else content.encode('utf8')
        self.send_response(status)
        self.send_header('content-type', content_type)
        self.send_header('content-length', str(len(body)))
        self.send_header('access-control-allow-origin', '*')
        self.end_headers()
        self.wfile.write(body)


def _writeln(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def main():
    gws.u.ensure_system_dirs()
    reload_config()

    httpd = http.server.ThreadingHTTPServer((HOST, PORT), HTTPRequestHandler)
    # shutdown() blocks until serve_forever acknowledges, which it cannot do
    # while the signal handler occupies the same thread. Exit instead. A handler
    # is still needed: as PID 1 in a container, a signal with the default
    # disposition is not delivered at all, and docker waits out its grace period.

    signal.signal(signal.SIGTERM, lambda x, y: sys.exit(0))
    _writeln(f'[mockmap] started on {HOST}:{PORT}, delay {DELAY}ms, config {_config_path}')
    httpd.serve_forever()


if __name__ == '__main__':
    main()

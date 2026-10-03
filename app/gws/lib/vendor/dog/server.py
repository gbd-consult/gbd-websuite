import os
import re
from typing import cast
import fnmatch
import threading
import http.server

import watchdog.observers
import watchdog.events

from . import util as u
from .builder import Builder
from .options import Options

RELOAD_SCRIPT_URL = '/___script'
RELOAD_TOKEN_URL = '/___reload'
RELOAD_POLL_SECONDS = 1.0

RELOAD_SCRIPT = (
    f"""
    const RELOAD_TOKEN_URL = "{RELOAD_TOKEN_URL}";
    const RELOAD_POLL_SECONDS = {RELOAD_POLL_SECONDS};
    """
    + """
(function () {
    let lastToken = null;
    let fn = () => {
        fetch(RELOAD_TOKEN_URL).then(r => r.text()).then(t => {
            if (lastToken === null) {
                lastToken = t;
            } else if (t !== lastToken) {
                location.reload();
            }
        });
    };
    setInterval(fn, RELOAD_POLL_SECONDS * 1000);
})();
"""
)

DEBOUNCE_SECONDS = 0.2

RESPONSE_HEADERS = [
    ('Cache-Control', 'must-revalidate, max-age=0, no-cache, no-store'),
    ('Expires', 'Tue, 01 Jan 1980 12:34:56 GMT'),
    ('Access-Control-Allow-Origin', '*'),
]


class Server:
    def __init__(self, opts: Options | dict):
        self.b: Builder = Builder(opts)
        self.extraPaths = self.collect_extra_paths()
        self.lock = threading.RLock()
        self.token = 0
        self._timer: threading.Timer | None = None
        self.changes = set()

    ## http

    def app(self, environ, start_response):
        path = environ['PATH_INFO']

        if path == RELOAD_SCRIPT_URL:
            return self.send(start_response, '200 OK', 'application/javascript', RELOAD_SCRIPT)

        if path == RELOAD_TOKEN_URL:
            with self.lock:
                token = str(self.token)
            return self.send(start_response, '200 OK', 'text/plain', token)

        url = path + 'index.html' if path.endswith('/') else path
        with self.lock:
            res = self.b.content_for_url(url)
        if not res:
            res = self.fallback_content(path)
        if not res:
            return self.send(start_response, '404 Not Found', 'text/html', 'Not Found')
        return self.send(start_response, '200 OK', res[0], res[1])

    def fallback_content(self, path):
        # Override this method to provide custom fallback content for a given path.
        # Return a tuple of (mime, body) or None if no fallback is available.
        return None

    def send(self, start_response, status, mime, body):
        if mime == 'text/html':
            if isinstance(body, bytes):
                body = body.decode('utf8')
            body = str(body) + f'\n<script src="{RELOAD_SCRIPT_URL}"></script>\n'
        body = body.encode('utf8') if isinstance(body, str) else bytes(body)
        headers = [('Content-Type', mime), ('Content-Length', str(len(body)))]
        start_response(status, headers + RESPONSE_HEADERS)
        return [body]

    ## watching

    def on_fs_event(self, event):
        if event.is_directory:
            return
        paths = []
        if self.watches(event.src_path):
            paths.append(event.src_path)
        dest = getattr(event, 'dest_path', None)
        if dest and self.watches(dest):
            paths.append(dest)
        if paths:
            u.log.debug(f'server: fs_event {event.event_type}: {", ".join(paths)}')
            self.changes.update(paths)
            self.schedule_rebuild()

    def collect_extra_paths(self):
        ps = []
        if self.b.pageTemplate:
            ps.append(self.b.pageTemplate)
        if self.b.includeTemplate:
            ps.append(self.b.includeTemplate)
        ps.extend(self.b.extraAssets)
        return ps

    def watches(self, path):
        path = os.path.abspath(path)
        if path in self.extraPaths:
            return True
        ex = self.b.options.excludeRegex
        if ex and re.search(ex, path):
            return False
        name = os.path.basename(path)
        patterns = self.b.options.docPatterns + self.b.options.assetPatterns
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    def init_watchers(self, observer):
        watcher = _Watcher(self)

        def subdir(r, s):
            return s == r or s.startswith(r + os.sep)

        roots = []
        for d in sorted(self.b.docRoots):
            if not any(subdir(r, d) for r in roots):
                roots.append(d)

        dirs = []
        for p in sorted(self.extraPaths):
            d = os.path.dirname(p)
            if d not in dirs and not any(subdir(r, d) for r in roots):
                dirs.append(d)

        for d in roots:
            observer.schedule(watcher, d, recursive=True)
            u.log.info(f'watching {d}')

        for d in dirs:
            observer.schedule(watcher, d, recursive=False)
            u.log.info(f'watching {d}')

    def schedule_rebuild(self):
        with self.lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self.rebuild)
            self._timer.start()

    def rebuild(self):
        with self.lock:
            cs = ', '.join(sorted(self.changes))
            self.changes = set()
            u.log.info(f'server: BEGIN rebuild' + (f' changes: {cs}' if cs else ''))
            self.b.build_html(write=False)
            self.token += 1
            u.log.info('server: END rebuild')

    ## lifecycle

    def start(self):
        self.rebuild()

        observer = watchdog.observers.Observer()
        self.init_watchers(observer)
        observer.start()

        host, port = self.b.options.serverHost, self.b.options.serverPort
        httpd = _HTTPServer((host, port), _Handler, self)

        u.log.info(f'http://{host}:{port}{self.b.options.webRoot}/')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()
            observer.join()
            httpd.server_close()


class _HTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, srv: Server):
        super().__init__(address, handler)
        self.srv = srv


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path, _, query = self.path.partition('?')
        environ = {'REQUEST_METHOD': 'GET', 'PATH_INFO': path, 'QUERY_STRING': query}
        app = cast(_HTTPServer, self.server).srv.app
        for chunk in app(environ, self._start_response):
            self.wfile.write(chunk)

    def _start_response(self, status, headers, exc_info=None):
        code, _, reason = status.partition(' ')
        self.send_response(int(code), reason or None)
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()

    def log_message(self, fmt, *args):
        u.log.debug('server: ' + (fmt % args))


class _Watcher(watchdog.events.FileSystemEventHandler):
    def __init__(self, srv: Server):
        self.srv = srv

    def on_created(self, event):
        self.srv.on_fs_event(event)

    def on_deleted(self, event):
        self.srv.on_fs_event(event)

    def on_modified(self, event):
        self.srv.on_fs_event(event)

    def on_moved(self, event):
        self.srv.on_fs_event(event)

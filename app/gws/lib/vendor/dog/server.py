import livereload

from . import builder, util


class Server:
    def __init__(self, options):
        self.options = util.to_data(options)
        self.liveServer = None
        self.liveScript = f'<script src="//{self.options.serverHost}:{self.options.serverPort}/livereload.js?port={self.options.serverPort}"></script>'
        self.builder = None

    def app(self, env, start_response):

        url = env['PATH_INFO']
        if url.endswith('/'):
            url += 'index.html'

        res = self.builder.content_for_url(url)
        if not res:
            start_response('404 Not Found', [('Content-type', 'text/html')])
            return [b'Not Found']

        mime, content = res
        if mime == 'text/html':
            content += self.liveScript

        if isinstance(content, str):
            content = content.encode('utf8')
        headers = [
            ('Content-type', mime),
            ('Cache-Control', ' must-revalidat, max-age=0, no-cache, no-store'),
            ('Expires', 'Tue, 01 Jan 1980 12:34:56 GMT'),
            ('Content-Length', len(content))
        ]
        start_response('200 OK', headers)
        return [content]

    def build(self):
        self.builder = builder.Builder(self.options)
        self.builder.build_all('html', write=False)

    def watch_docs(self, args=None):
        util.log.debug(f'watch_docs: {args!r}')
        if args:
            self.build()

    def watch_assets(self, args=None):
        util.log.debug(f'watch_assets: {args!r}')

    def start(self):
        self.build()

        self.liveServer = livereload.Server(self.app)

        for root in self.options.rootDirs:
            for p in self.options.docPatterns:
                path = root + '/**/' + p
                self.liveServer.watch(path, self.watch_docs, delay=0.1)
                util.log.info(f'watching {path}')

        for path in self.builder.assetPaths:
            self.liveServer.watch(path, self.watch_assets, delay=0.1)

        for path in self.builder.options.extraAssets:
            self.liveServer.watch(path, self.watch_assets, delay=0.1)

        try:
            self.liveServer.setHeader('Access-Control-Allow-Origin', '*')
            self.liveServer.setHeader('Access-Control-Allow-Methods', '*')
        except AttributeError:
            pass

        # hack around https://github.com/lepture/python-livereload/issues/176
        import tornado.autoreload
        class ListNoAppend(list):
            def append(self, x):
                pass

        tornado.autoreload._reload_hooks = ListNoAppend()

        util.log.info(f'http://{self.options.serverHost}:{self.options.serverPort}{self.options.webRoot}/')

        self.liveServer.serve(
            host=self.options.serverHost,
            port=self.options.serverPort,
            debug=True,
            restart_delay=1,
            open_url_delay=None,
        )
"""Actions CLI"""

import cProfile

import gws
import gws.config
import gws.base.action
import gws.base.auth
import gws.base.web
import gws.lib.jsonx
import gws.lib.vendor.slon


class InvokeRequest(gws.Request):
    cmd: str
    params: str


class Object(gws.Node):

    @gws.ext.command.cli('actionInvoke')
    def invoke(self, p: InvokeRequest):
        """Invoke a web action."""

        res = self._invoke(p)
        print(gws.lib.jsonx.to_pretty_string(res))

    @gws.ext.command.cli('actionProfile')
    def profile(self, p: InvokeRequest):
        """Profile a web action."""

        filename = f'{gws.c.VAR_DIR}/{p.cmd}.pstats'

        cProfile.runctx(
            'self._invoke(p)',
            {},
            locals(),
            filename=filename
        )
        print(f'profile saved to {filename!r}')

    def _invoke(self, p: InvokeRequest):
        environ = {}
        root = gws.config.load()
        site = root.app.webMgr.site_from_environ(environ)
        req = gws.base.web.wsgi.Requester(root, environ, site)

        fn, request = root.app.actionMgr.prepare_action(
            gws.CommandCategory.api,
            p.cmd,
            gws.lib.vendor.slon.parse(p.params),
            req.user
        )
        try:
            return fn(req, request)
        except:
            gws.log.exception()

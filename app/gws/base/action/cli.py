"""CLI commands to invoke and profile web actions."""

import cProfile

import gws
import gws.base.action
import gws.base.auth
import gws.base.web
import gws.lib.jsonx
import gws.lib.vendor.slon


class InvokeRequest(gws.Request):
    """Parameters of the ``actionInvoke`` and ``actionProfile`` CLI commands."""

    cmd: str
    """API command name, for example ``mapDescribeLayer``."""
    params: str
    """Command parameters in SLON format."""


class Object(gws.Node):
    """CLI commands for web actions.

    Invokes an API command from the command line and prints the response, or runs
    it with the profiler and saves the statistics.
    """

    @gws.ext.command.cli('actionInvoke')
    def invoke(self, p: InvokeRequest):
        """Invoke an API command and print the response as JSON."""

        res = self._invoke(p)
        print(gws.lib.jsonx.to_pretty_string(res))

    @gws.ext.command.cli('actionProfile')
    def profile(self, p: InvokeRequest):
        """Run an API command with the profiler and save the statistics."""

        filename = f'{gws.c.VAR_DIR}/{p.cmd}.pstats'

        cProfile.runctx(
            'self._invoke(p)',
            {},
            locals(),
            filename=filename
        )
        print(f'profile saved to {filename!r}')

    def _invoke(self, p: InvokeRequest):
        """Load the configuration, run an API command with an empty web request and return its response."""
        environ = {}
        root = gws.load_root()
        req = gws.base.web.wsgi.Requester(root, environ, root.app.webMgr.site)

        fn, request = root.app.actionMgr.prepare_action(
            gws.CommandCategory.api,
            p.cmd,
            gws.lib.vendor.slon.parse(p.params),
            '',
            req.user,
        )
        try:
            return fn(req, request)
        except:
            gws.log.exception()

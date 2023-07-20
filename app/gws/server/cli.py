"""Command-line server commands."""

import gws
import gws.types as t

from . import control


class Params(gws.CliParams):
    config: t.Optional[str]
    """configuration file"""
    manifest: t.Optional[str]
    """manifest file"""


gws.ext.new.cli('server')


class Object(gws.Node):

    @gws.ext.command.cli('serverStart')
    def do_start(self, p: Params):
        """Configure and start the server."""

        self._setenv(p)
        control.start(p.manifest, p.config)

    @gws.ext.command.cli('serverReload')
    def do_reload(self, p: Params):
        """Restart the server."""

        self._setenv(p)
        control.reload_all()

    @gws.ext.command.cli('serverReconfigure')
    def do_reconfigure(self, p: Params):
        """Reconfigure and restart the server."""

        self._setenv(p)
        control.reconfigure(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigure')
    def do_configure(self, p: Params):
        """Configure the server, but do not restart."""

        self._setenv(p)
        control.configure_and_store(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigtest')
    def do_configtest(self, p: Params):
        """Test the configuration."""

        self._setenv(p)
        control.configure(p.manifest, p.config)

    def _setenv(self, p: Params):
        p.config = p.config or gws.env.GWS_CONFIG
        p.manifest = p.manifest or gws.env.GWS_MANIFEST

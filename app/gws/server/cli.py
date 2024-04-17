"""Command-line server commands."""

from typing import Optional

import gws

from . import control


class Params(gws.CliParams):
    config: Optional[str]
    """configuration file"""
    manifest: Optional[str]
    """manifest file"""


gws.ext.new.cli('server')


class Object(gws.Node):

    @gws.ext.command.cli('serverStart')
    def do_start(self, p: Params):
        """Configure and start the server."""

        control.start(p.manifest, p.config)

    @gws.ext.command.cli('serverReload')
    def do_reload(self, p: Params):
        """Restart the server."""

        control.reload_all()

    @gws.ext.command.cli('serverReconfigure')
    def do_reconfigure(self, p: Params):
        """Reconfigure and restart the server."""

        control.reconfigure(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigure')
    def do_configure(self, p: Params):
        """Configure the server, but do not restart."""

        control.configure_and_store(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigtest')
    def do_configtest(self, p: Params):
        """Test the configuration."""

        control.configure(p.manifest, p.config)

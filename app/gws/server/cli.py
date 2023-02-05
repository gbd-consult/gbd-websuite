"""Command-line server commands."""

import gws
import gws.types as t

from . import control


class StartParams(gws.CliParams):
    config: t.Optional[str]
    """configuration file"""
    manifest: t.Optional[str]
    """manifest file"""


class ReloadParams(StartParams):
    modules: t.Optional[t.List[str]]
    """list of modules to reload ('qgis', 'mapproxy', 'web', 'spool')"""


gws.ext.new.cli('server')

class Object(gws.Node):

    @gws.ext.command.cli('serverStart')
    def do_start(self, p: StartParams):
        """Configure and start the server"""
        control.start(p.manifest, p.config)

    @gws.ext.command.cli('serverRestart')
    def do_restart(self, p: StartParams):
        """Stop and start the server"""
        self.do_start(p)

    @gws.ext.command.cli('serverStop')
    def do_stop(self, p: gws.EmptyRequest):
        """Stop the server"""
        control.stop()

    @gws.ext.command.cli('serverReload')
    def do_reload(self, p: ReloadParams):
        """Reload specific (or all) server modules"""
        if not control.reload(p.modules):
            gws.log.info('server not running, starting')
            self.do_start(t.cast(StartParams, p))

    @gws.ext.command.cli('serverReconfigure')
    def do_reconfigure(self, p: StartParams):
        """Reconfigure and restart the server"""
        control.reconfigure(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigure')
    def do_configure(self, p: StartParams):
        """Configure the server, but do not restart"""
        control.configure_and_store(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigtest')
    def do_configtest(self, p: StartParams):
        """Test the configuration"""
        control.configure(p.manifest or '', p.config or '', is_starting=False)

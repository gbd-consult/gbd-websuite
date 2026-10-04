"""Command-line server commands."""

from typing import Optional

import gws

from . import control


class Params(gws.CliParams):
    """Parameters for server commands."""
    config: Optional[str]
    """Configuration file."""
    manifest: Optional[str]
    """Manifest file."""


class ConfigTestParams(gws.CliParams):
    """Parameters for the config test command."""
    config: Optional[str]
    """Configuration file."""
    manifest: Optional[str]
    """Manifest file."""
    dirs: Optional[str]
    """Directories to watch for changes, ``/data`` by default."""
    watch: Optional[bool]
    """Repeat the test whenever a file in the watched directories changes."""
    parse: Optional[bool]
    """Only parse the configuration, do not configure the objects."""


@gws.ext.object.cli('server')
class Object(gws.Node):
    """Command-line commands for starting, reloading and configuring the server."""

    @gws.ext.command.cli('serverStart')
    def do_start(self, p: Params):
        """Configure and start the server."""

        control.start(p.manifest, p.config)

    @gws.ext.command.cli('serverReload')
    def do_reload(self, p: Params):
        """Reload the server without reconfiguring."""

        control.reload_all()

    @gws.ext.command.cli('serverReconfigure')
    def do_reconfigure(self, p: Params):
        """Reconfigure and reload the server."""

        control.reconfigure(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigure')
    def do_configure(self, p: Params):
        """Configure the server without reloading."""

        control.configure_and_store(p.manifest, p.config)

    @gws.ext.command.cli('serverConfigtest')
    def do_configtest(self, p: ConfigTestParams):
        """Test the configuration and report errors, optionally watching for changes."""

        control.config_test(
            p.manifest,
            p.config,
            p.dirs,
            with_parse_only=p.parse,
            with_watch=p.watch,
        )

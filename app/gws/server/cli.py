from argh import arg

import gws
from . import control

COMMAND = 'server'


@arg('--cfg', help='configuration file')
def start(cfg=None):
    """Create the server start script."""

    control.start(cfg)


def stop():
    """Stop the server"""

    control.stop()


@arg('--cfg', help='configuration file')
def reconfigure(cfg=None):
    """Reconfigure and gracefully reload the server"""

    control.reconfigure(cfg)


@arg('--modules', help='server modules to reload')
def reload(modules=None):
    """Gracefully reload the server without reconfiguring"""

    control.reload(gws.as_list(modules))


@arg('--cfg', help='configuration file')
def configure(cfg=None):
    """Configure the server"""

    control.configure(cfg)

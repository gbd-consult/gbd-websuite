from argh import arg
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
def reload(cfg=None):
    """Reconfigure and gracefully reload the server"""

    control.reload(cfg)


@arg('--module', help='server module to reset')
def reset(module=None):
    """Gracefully reload the server without reconfiguring"""

    control.reset(module)


@arg('--cfg', help='configuration file')
def configure(cfg=None):
    """Configure the server"""

    control.configure(cfg)

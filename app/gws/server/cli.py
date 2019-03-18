from argh import arg
from . import control

COMMAND = 'server'


def start():
    """Start the server"""

    control.start()


def stop():
    """Stop the server"""

    control.stop()


def reload():
    """Reconfigure and gracefully reload the server"""

    control.reload()


def reset():
    """Gracefully reload the server without reconfiguring"""

    control.reset()


@arg('--module', help='server module to reset')
def reset(module=None):
    """Gracefully reload the server without reconfiguring"""

    control.reset(module)


def configure():
    """Configure the server"""

    control.configure()

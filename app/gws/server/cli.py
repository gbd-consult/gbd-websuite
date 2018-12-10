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


def configure():
    """Configure the server"""

    control.configure()

import gws
import gws.types as t
import gws.config
import gws.config.loader
import gws.lib.clihelpers as clihelpers
import gws.lib.json2

COMMAND = 'dprocon'


def setup():
    """Create Dprocon index tables"""

    gws.config.loader.load()
    action = clihelpers.find_action('dprocon')
    if action:
        action.setup()

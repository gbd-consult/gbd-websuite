import gws
import gws.config
import gws.config.loader
import gws.lib.json2
import gws.lib.clihelpers as clihelpers

COMMAND = 'dprocon'


def setup():
    """Create Dprocon index tables"""

    gws.config.loader.load()
    action = clihelpers.find_action('dprocon')
    if action:
        action.setup()

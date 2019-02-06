import gws
import gws.config
import gws.config.loader
import gws.tools.json2
import gws.tools.clihelpers as clihelpers

COMMAND = 'dprocon'


def setup():
    """Create Dprocon index tables"""

    gws.config.loader.load()
    action = clihelpers.find_action('dprocon')
    if action:
        action.setup()

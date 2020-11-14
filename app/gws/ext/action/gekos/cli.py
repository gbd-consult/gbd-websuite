from argh import arg
import gws.tools.clihelpers as clihelpers

COMMAND = 'gekos'


@arg('--project', help='project unique ID')
def setup(project=None):
    """Load the data from Gekos-Online into a postgres table"""

    a = clihelpers.find_action('gekos', project)
    if a:
        a.load_data()

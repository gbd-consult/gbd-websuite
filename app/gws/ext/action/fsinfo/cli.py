from argh import arg
import gws.lib.clihelpers as clihelpers

COMMAND = 'fsinfo'

@arg('--project', help='project unique ID')
@arg('--path', help='base directory')
def read(project=None, path=None):
    """Import PDFs from the base dir"""

    action = clihelpers.find_action('fsinfo', project)
    if action:
        action.do_read(path)

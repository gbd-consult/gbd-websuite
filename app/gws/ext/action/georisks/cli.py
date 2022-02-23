from argh import arg

import gws.tools.clihelpers as clihelpers
import gws.tools.misc



COMMAND = 'georisks'


@arg('--dir', help='directory to export to')
@arg('--project', help='project unique ID')
def export(dir=None, project=None):
    """Export reports."""

    action = clihelpers.find_action('georisks', project)
    gws.ensure_dir(dir)
    n = action.export_reports(dir)
    print(f'{n} report(s) saved')

@arg('--project', help='project unique ID')
def aartelink(dir=None, project=None):
    """aartelink service request."""

    action = clihelpers.find_action('georisks', project)
    action.aartelink_service()

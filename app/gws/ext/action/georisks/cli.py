import time

from argh import arg

import gws
import gws.config
import gws.config.loader
import gws.tools.clihelpers as clihelpers
import gws.tools.json2
import gws.tools.net



COMMAND = 'georisks'


@arg('--dir', help='directory to export to')
@arg('--project', help='project unique ID')
def export(dir=None, project=None):

    action = clihelpers.find_action('georisks', project)
    n = action.export_reports(dir)
    print(f'{n} report(s) saved')

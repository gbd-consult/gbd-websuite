from argh import arg

import gws.lib.clihelpers as clihelpers

COMMAND = 'bplan'


@arg('--project', help='project unique ID')
@arg('--path', help='zip file or directory path')
@arg('--replace', help='replace the ags (area) completely')
def read(project=None, path=None, replace=False):
    """Read and parse bplan source files."""

    action = clihelpers.find_action('bplan', project)
    if action:
        action.do_import(path, replace)


@arg('--project', help='project unique ID')
def update(project=None):
    """Update bplan VRT and Qgis files."""

    action = clihelpers.find_action('bplan', project)
    if action:
        action.do_update()

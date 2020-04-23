from argh import arg

import gws.tools.clihelpers as clihelpers

COMMAND = 'bplan'


@arg('--path', help='zip file or directory path')
@arg('--replace', help='replace the ags (area) completely')
def read(path=None, replace=False):
    """Read and parse bplan source files."""

    action = clihelpers.find_action('bplan')
    if action:
        action.do_import(path, replace)


def update():
    """Update bplan VRT and Qgis files."""

    action = clihelpers.find_action('bplan')
    if action:
        action.do_update()

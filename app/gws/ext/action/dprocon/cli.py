import getpass
import os

from argh import arg
import time
import gws
import gws.config
import gws.config.loader
import gws.tools.json2
import gws.tools.clihelpers as ch

COMMAND = 'dprocon'


def create_index():
    """Create Dprocon index tables"""

    gws.config.loader.load()
    a = gws.config.find_first('gws.ext.action.dprocon')
    if a:
        user, password = ch.database_credentials()
        a.create_index(user, password)


def setup():
    """Initialize Dprocon helper tables"""

    gws.config.loader.load()
    a = gws.config.find_first('gws.ext.action.dprocon')
    if a:
        user, password = ch.database_credentials()
        a.create_index(user, password)
        a.create_tables(user, password)

import getpass
import os

from argh import arg
import time
import gws
import gws.config
import gws.config.loader
import gws.tools.json2
from .tools import nas

COMMAND = 'alkis'


@arg('--path', help='path to the NAS zip archive')
def parse(path=None):
    """Preprocess the NAS data model files"""

    # 'NAS_6.0.zip' can be downloaded from
    # see http://www.adv-online.de/AAA-Modell/Dokumente-der-GeoInfoDok/GeoInfoDok-6.0/
    # under "Das externe Modell, Datenaustausch"

    props = nas.parse_properties(path)
    print(gws.tools.json2.to_string(props, pretty=True))


@arg('--project', help='project unique ID')
def create_index(project=None):
    """Create an internal ALKIS search index for a project"""

    a = _alkis_action(project)
    if a:
        user, password = _get_credentials()
        t = time.time()
        a.index_create(user, password)
        t = time.time() - t
        gws.log.info('index done in %.2f sec' % t)


@arg('--project', help='project unique ID')
def check_index(project=None):
    """Check the status of the ALKIS search index"""

    a = _alkis_action(project)
    if a:
        if a.index_ok():
            gws.log.info(f'ALKIS indexes in {project} are ok')
        else:
            gws.log.info(f'ALKIS indexes in {project} are NOT ok')


@arg('--project', help='project unique ID')
def drop_index(project=None):
    """Remove the ALKIS search index"""

    a = _alkis_action(project)
    if a:
        user, password = _get_credentials()
        a.index_drop(user, password)


def _alkis_action(project):
    gws.config.loader.load()

    if project:
        loc = gws.config.find('gws.common.project', gws.as_uid(project))
        if not loc:
            gws.log.error('project not found')
            return
    else:
        loc = gws.config.find_all('gws.common.application')[0]

    a = loc.action('alkis')
    if not a:
        gws.log.error('ALKIS action not configured')
        return

    return a


def _get_credentials():
    if 'PGUSER' in os.environ and 'PGPASSWORD' in os.environ:
        return os.environ['PGUSER'], os.environ['PGPASSWORD']

    user = input('DB Username: ')
    password = getpass.getpass('Password   : ')
    return user, password

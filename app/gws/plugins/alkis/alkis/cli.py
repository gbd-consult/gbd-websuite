import time

from argh import arg

import gws
import gws.config
import gws.config.loader
import gws.lib.clihelpers as clihelpers
import gws.lib.json2

import gws.ext.helper.alkis as alkis

import gws.types as t

from .util import nas

COMMAND = 'alkis'


@arg('--path', help='path to the NAS zip archive')
def parse(path=None):
    """Preprocess the NAS data model files"""

    # 'NAS_6.0.zip' can be downloaded from
    # see http://www.adv-online.de/AAA-Modell/Dokumente-der-GeoInfoDok/GeoInfoDok-6.0/
    # under "Das externe Modell, Datenaustausch"

    props = nas.parse_properties(path)
    print(gws.lib.json2.to_string(props, pretty=True))


def setup():
    """Create an internal ALKIS search index."""

    a = _get_alkis()
    if a:
        user, password = clihelpers.database_credentials()
        ts = time.time()
        a.create_index(user, password)
        t = time.time() - ts
        gws.log.info('index done in %.2f sec' % t)


def create_index():
    """Create an internal ALKIS search index."""

    setup()


def check_index():
    """Check the status of the ALKIS search index."""

    a = _get_alkis()
    if a:
        if a.index_ok():
            gws.log.info(f'ALKIS indexes are ok')
        else:
            gws.log.info(f'ALKIS indexes are NOT ok')


def drop_index():
    """Remove the ALKIS search index"""

    a = _get_alkis()
    if a:
        user, password = clihelpers.database_credentials()
        a.drop_index(user, password)


def _get_alkis() -> t.Optional[alkis.Object]:
    root = gws.config.loader.load()

    a = t.cast(alkis.Object, root.find_first('gws.ext.helper.alkis'))
    if not a:
        gws.log.error('ALKIS helper is not configured')
        return
    if not a.has_source:
        gws.log.error('ALKIS source data not found')
        return
    return a

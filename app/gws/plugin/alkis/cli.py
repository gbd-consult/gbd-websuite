import getpass
import os
import time

import gws
import gws.config
import gws.lib.jsonx
import gws.types as t

from . import provider, search
from .util import nas


class ParseParams(gws.CliParams):
    path: str 
    """path to the NAS zip archive"""


class IndexParams(gws.CliParams):
    uid: t.Optional[str] 
    """alkis action uid"""


@gws.ext.object.cli('alkis')
class Object(gws.Node):

    @gws.ext.command.cli('alkisParse')
    def parse(self, p: ParseParams):
        """Preprocess the NAS data model files"""

        # 'NAS_6.0.zip' can be downloaded from
        # see http://www.adv-online.de/AAA-Modell/Dokumente-der-GeoInfoDok/GeoInfoDok-6.0/
        # under "Das externe Modell, Datenaustausch"

        props = nas.parse_properties(p.path)
        print(gws.lib.jsonx.to_string(props, pretty=True))

    @gws.ext.command.cli('alkisSetup')
    def setup(self, p: IndexParams):
        """Create the ALKIS search index"""

        prov = self._get_provider(p.uid)
        if not prov:
            return

        user, password = self._database_credentials()
        ts = time.time()
        prov.create_index(user, password)
        ts = time.time() - ts
        gws.log.info(f'index done in {ts:.2f} sec')

    @gws.ext.command.cli('alkisDrop')
    def drop(self, p: IndexParams):
        """Remove the ALKIS search index"""

        prov = self._get_provider(p.uid)
        if not prov:
            return

        user, password = self._database_credentials()
        prov.drop_index(user, password)

    @gws.ext.command.cli('alkisCheck')
    def check(self, p: IndexParams):
        """Check the status of the ALKIS search index."""

        prov = self._get_provider(p.uid)
        if prov:
            gws.log.info(f"Database     : {prov.connect_params.get('database')}")
            gws.log.info(f'CRS          : {prov.crs}')
            gws.log.info(f'Data schema  : {prov.data_schema}')
            gws.log.info(f'Index schema : {prov.index_schema}')

            if prov.index_ok():
                gws.log.info(f'ALKIS indexes are ok')
            else:
                gws.log.info(f'ALKIS indexes are NOT ok')

    def _database_credentials(self):
        if 'PGUSER' in os.environ and 'PGPASSWORD' in os.environ:
            return os.environ['PGUSER'], os.environ['PGPASSWORD']

        user = input('DB username: ')
        password = getpass.getpass('DB password: ')
        return user, password

    def _get_provider(self, uid) -> t.Optional[provider.Object]:
        root = gws.config.load()
        action = root.find(uid=uid) if uid else root.find(klass='gws.ext.action.alkissearch')
        if not action:
            gws.log.error('no ALKIS action found')
            return None
        return t.cast(search.Object, action).provider

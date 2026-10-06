"""The ``gekos`` action."""

from typing import Optional, cast

import gws
import gws.lib.crs
import gws.base.feature
import gws.lib.shape
import gws.base.action
import gws.base.database

import gws.plugin.alkis.action as alkis_action

from . import core, index


class GetXyRequest(gws.Request):
    """Request for the coordinates of a parcel or an address."""

    fs: Optional[str]
    """Combined parcel (Flurstueck) code."""
    ad: Optional[str]
    """Combined address code."""


@gws.ext.config.action('gekos')
class Config(gws.base.action.Config):
    """Integration with the GekoS-Bau software."""

    index: Optional[core.IndexConfig]
    """Index of GekoS records loaded from gek-online."""


@gws.ext.object.action('gekos')
class Object(gws.base.action.Object):
    """GekoS action."""

    idx: index.Object
    """The GekoS index, or ``None`` if not configured."""

    def configure(self):
        self.idx = self.create_child_if_configured(index.Object, self.cfg('index'))

    @gws.ext.command.get('gekosGetXY')
    def get_xy(self, req: gws.WebRequester, p: GetXyRequest) -> gws.ContentResponse:
        """Return the coordinates of a parcel or an address."""
        project = None
        if p.projectUid:
            project = req.user.require_project(p.projectUid)

        alkis = cast(alkis_action.Object, self.root.app.actionMgr.find_action(project, 'alkis', req.user))
        if not alkis:
            gws.log.error(f'gekos: alkis action not found, {p.projectUid=}')
            return gws.ContentResponse(mimeType='text/plain', content='error:')

        lst = None
        if p.fs:
            lst, _ = alkis.find_flurstueck_objects(req, alkis_action.FindFlurstueckRequest(combinedFlurstueckCode=p.fs))
        elif p.ad:
            lst, _ = alkis.find_adresse_objects(req, alkis_action.FindAdresseRequest(combinedAdresseCode=p.ad))

        if not lst:
            gws.log.error(f'gekos: not found, {p.fs=} {p.ad=}')
            return gws.ContentResponse(mimeType='text/plain', content='error:')

        return gws.ContentResponse(mimeType='text/plain', content='{:.3f};{:.3f}'.format(lst[0].x, lst[0].y))

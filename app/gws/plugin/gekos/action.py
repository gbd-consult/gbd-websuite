"""GekoS action."""

from typing import Optional, cast

import gws
import gws.lib.crs
import gws.base.feature
import gws.base.shape
import gws.base.action
import gws.base.database

import gws.plugin.alkis.action as alkis_action

from . import core, index

gws.ext.new.action('gekos')


class GetXyRequest(gws.Request):
    """Request to get XY coordinates from a GekoS feature."""

    fs: Optional[str]
    """Combined flurstueck code."""
    ad: Optional[str]
    """Combined adresse code."""


class GetFsResponse(gws.Response):
    feature: gws.FeatureProps


class Config(gws.base.action.Config):
    """GekoS action configuration."""

    index: Optional[core.IndexConfig]
    """GekoS index configuration."""
    templates: Optional[list[gws.ext.config.template]]
    """Feature templates."""


_DEFAULT_TEMPLATES = [
    gws.Config(subject='feature.title', type='html', text='{vollnummer}'),
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='Flurstück {vollnummer}',
    ),
]


class Object(gws.base.action.Object):
    idx: index.Object
    templates: list[gws.Template]

    def configure(self):
        self.idx = self.create_child_if_configured(index.Object, self.cfg('index'))
        p = self.cfg('templates', default=[]) + _DEFAULT_TEMPLATES
        self.templates = [self.create_child(gws.ext.object.template, c) for c in p]

    @gws.ext.command.get('gekosGetXY')
    def get_xy(self, req: gws.WebRequester, p: GetXyRequest) -> gws.ContentResponse:
        project = None
        if p.projectUid:
            project = req.user.require_project(p.projectUid)

        alkis = cast(alkis_action.Object, self.root.app.actionMgr.find_action(project, 'alkis', req.user))
        if not alkis:
            gws.log.error(f'gekos: alkis action not found, {p.projectUid=}')
            return gws.ContentResponse(mime='text/plain', content='error:')

        lst = None
        if p.fs:
            lst, _ = alkis.find_flurstueck_objects(req, alkis_action.FindFlurstueckRequest(combinedFlurstueckCode=p.fs))
        elif p.ad:
            lst, _ = alkis.find_adresse_objects(req, alkis_action.FindAdresseRequest(combinedAdresseCode=p.ad))

        if not lst:
            gws.log.error(f'gekos: not found, {p.fs=} {p.ad=}')
            return gws.ContentResponse(mime='text/plain', content='error:')

        return gws.ContentResponse(mime='text/plain', content='{:.3f};{:.3f}'.format(lst[0].x, lst[0].y))

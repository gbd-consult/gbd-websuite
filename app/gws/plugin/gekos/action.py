"""Interface with GekoS-Bau software."""

from typing import Optional, cast

import gws
import gws.lib.crs
import gws.base.feature
import gws.base.shape
import gws.base.action
import gws.base.database

import gws.plugin.alkis.action as alkis_action

from . import index

gws.ext.new.action('gekos')

"""

    see https://www.gekos.de/
    
    GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)
    
    base address:
    
        GIS-URL-Base  = http://my-server

    client-side call, handled in the client by the Marker element
    
        GIS-URL-ShowXY  = /project/PROJECT_ID/?x=<x>&y=<y>&z=SCALE_VALUE
        
    client-side call, handled in the client by js/index.tsx
        
        GIS-URL-GetXYFromMap = /project/PROJECT_ID/?&x=<x>&y=<y>&gekosUrl=<returl>
    
    client-side call, handled in the Alkis plugin
    
        GIS-URL-ShowFs = /project/PROJECT_ID/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        
    callback urls, handled here
            
        GIS-URL-GetXYFromFs   = /_/gekosGetXY/projectUid/PROJECT_ID/fs/<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
        GIS-URL-GetXYFromGrd  = /_/gekosGetXY/projectUid/PROJECT_ID/ad/<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

    NB: the order of placeholders must match COMBINED_FLURSTUECK_FIELDS and COMBINED_ADRESSE_FIELDS in the Alkis Plugin

"""


class GetXyRequest(gws.Request):
    fs: Optional[str]
    ad: Optional[str]


class GetFsResponse(gws.Response):
    feature: gws.FeatureProps


class Config(gws.base.action.Config):
    """GekoS action"""

    index: Optional[index.Config]
    """GekoS index configuration"""
    templates: Optional[list[gws.ext.config.template]]
    """feature templates"""


_DEFAULT_TEMPLATES = [
    gws.Config(
        subject='feature.title',
        type='html',
        text='{vollnummer}'
    ),
    gws.Config(
        subject='feature.teaser',
        type='html',
        text='Flurstück {vollnummer}',
    )
]


class Object(gws.base.action.Object):
    index: Optional[index.Object]
    templates: list[gws.Template]

    def configure(self):
        self.index = self.create_child_if_configured(index.Object, self.cfg('index'))
        p = self.cfg('templates', default=[]) + _DEFAULT_TEMPLATES
        self.templates = [self.create_child(gws.ext.object.template, c) for c in p]

    @gws.ext.command.get('gekosGetXY')
    def get_xy(self, req: gws.WebRequester, p: GetXyRequest) -> gws.ContentResponse:
        project = None
        if p.projectUid:
            project = req.user.require_project(p.projectUid)

        alkis = cast(
            alkis_action.Object,
            self.root.app.actionMgr.find_action(project, 'alkis', req.user)
        )
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

        return gws.ContentResponse(
            mime='text/plain',
            content='{:.3f};{:.3f}'.format(lst[0].x, lst[0].y))

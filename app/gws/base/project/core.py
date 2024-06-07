from typing import Optional

import gws
import gws.base.action
import gws.base.client
import gws.base.map
import gws.base.printer
import gws.base.template
import gws.base.web
import gws.lib.metadata

gws.ext.new.project('default')


class Config(gws.ConfigWithAccess):
    """Project configuration"""

    type: str = 'default'

    actions: Optional[list[gws.ext.config.action]]
    """project-specific actions"""
    assets: Optional[gws.base.web.site.WebDocumentRootConfig]
    """project-specific assets options"""
    client: Optional[gws.base.client.Config]
    """project-specific gws client configuration"""
    finders: Optional[list[gws.ext.config.finder]]
    """search providers"""
    locales: Optional[list[str]]
    """project locales"""
    map: Optional[gws.base.map.Config]
    """Map configuration"""
    metadata: Optional[gws.Metadata]
    """project metadata"""
    models: Optional[list[gws.ext.config.model]]
    """data models"""
    overviewMap: Optional[gws.base.map.Config]
    """Overview map configuration"""
    owsServices: Optional[list[gws.ext.config.owsService]]
    """OWS services configuration"""
    printers: Optional[list[gws.base.printer.Config]]
    """print configurations"""
    templates: Optional[list[gws.ext.config.template]]
    """project info templates"""
    title: str = ''
    """project title"""


class Props(gws.Props):
    actions: list[gws.ext.props.action]
    client: Optional[gws.base.client.Props]
    description: str
    locales: list[str]
    map: gws.ext.props.map
    models: list[gws.ext.props.model]
    metadata: gws.lib.metadata.Props
    overviewMap: gws.ext.props.map
    printers: list[gws.base.printer.Props]
    title: str
    uid: str


class Object(gws.Project):
    overviewMap: gws.base.map.Object
    title: str

    def configure(self):
        gws.log.info(f'configuring project {self.uid!r}')

        self.metadata = gws.lib.metadata.merge(
            self.root.app.metadata,
            gws.lib.metadata.from_config(self.cfg('metadata')))

        title = self.cfg('title') or self.metadata.get('title') or self.cfg('uid')
        # title at the top level config preferred to metadata
        self.title = self.metadata.title = title

        p = self.cfg('assets')
        self.assetsRoot = gws.WebDocumentRoot(p) if p else None

        self.localeUids = self.cfg('locales') or self.root.app.localeUids

        self.actions = self.create_children(gws.ext.object.action, self.cfg('actions'))
        self.map = self.create_child_if_configured(gws.ext.object.map, self.cfg('map'), _defaultTitle=self.title)
        self.overviewMap = self.create_child_if_configured(gws.base.map.Object, self.cfg('overviewMap'))
        self.printers = self.create_children(gws.ext.object.printer, self.cfg('printers'))
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
        self.finders = self.create_children(gws.ext.object.finder, self.cfg('finders'))
        self.templates = self.create_children(gws.ext.object.template, self.cfg('templates'))
        self.client = self.create_child_if_configured(gws.base.client.Object, self.cfg('client'))
        self.owsServices = self.create_children(gws.ext.object.owsService, self.cfg('owsServices'))

    def props(self, user):
        desc = None
        tpl = self.root.app.templateMgr.find_template(self, user=user, subject='project.description')
        if tpl:
            desc = tpl.render(gws.TemplateRenderInput(args={'project': self}, user=user))

        return gws.Props(
            actions=self.root.app.actionMgr.actions_for_project(self, user),
            client=self.client or self.root.app.client,
            description=desc.content if desc else '',
            map=self.map,
            metadata=gws.lib.metadata.props(self.metadata),
            models=[],
            overviewMap=self.overviewMap,
            printers=self.root.app.printerMgr.printers_for_project(self, user),
            title=self.title,
            uid=self.uid,
        )

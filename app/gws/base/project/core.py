from typing import Optional

import gws
import gws.base.action
import gws.base.client
import gws.base.map
import gws.base.printer
import gws.base.template
import gws.base.web
import gws.base.metadata


@gws.ext.config.project('default')
class Config(gws.ConfigWithAccess):
    """Project with its own map, actions, templates and services."""

    type: str = 'default'

    actions: Optional[list[gws.ext.config.action]]
    """Actions available only in this project."""
    assets: Optional[gws.base.web.site.WebDirConfig]
    """Directory with project assets, checked before the global assets directory."""
    client: Optional[gws.base.client.Config]
    """Client UI for this project, replacing the application client."""
    exporters: Optional[list[gws.ext.config.exporter]]
    """Exporters for this project, offered along with the global ones."""
    finders: Optional[list[gws.ext.config.finder]]
    """Search providers available in this project."""
    locales: Optional[list[gws.LocaleUid]]
    """Locales for this project."""
    map: Optional[gws.base.map.Config]
    """Main map of the project."""
    metadata: Optional[gws.base.metadata.Config]
    """Project metadata, merged with the application metadata."""
    models: Optional[list[gws.ext.config.model]]
    """Data models."""
    overviewMap: Optional[gws.base.map.Config]
    """Overview map, showing the current view on a smaller scale."""
    owsServices: Optional[list[gws.ext.config.owsService]]
    """OWS services published for this project."""
    printers: Optional[list[gws.base.printer.Config]]
    """Printers for this project, offered along with the global ones."""
    templates: Optional[list[gws.ext.config.template]]
    """Templates for this project."""
    title: str = ''
    """Project title."""
    vars: Optional[dict]
    """Custom variables."""


@gws.ext.props.project('default')
class Props(gws.Props):
    actions: list[gws.ext.props.action]
    client: Optional[gws.base.client.Props]
    description: str
    locales: list[str]
    map: gws.ext.props.map
    exporters: list[gws.ext.props.exporter]
    models: list[gws.ext.props.model]
    metadata: gws.base.metadata.Props
    overviewMap: gws.ext.props.map
    printers: list[gws.base.printer.Props]
    title: str
    uid: str


@gws.ext.object.project('default')
class Object(gws.Project):
    overviewMap: gws.base.map.Object
    title: str

    def configure(self):
        gws.log.info(f'configuring project {self.uid!r}')

        self.vars = self.cfg('vars') or {}

        self.metadata = gws.base.metadata.from_args(
            self.root.app.metadata,
            self.cfg('metadata'),
        )

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
        self.exporters = self.create_children(gws.ext.object.exporter, self.cfg('exporters'))
        self.client = self.create_child_if_configured(gws.base.client.Object, self.cfg('client'))
        self.owsServices = self.create_children(gws.ext.object.owsService, self.cfg('owsServices'))

    def props(self, user):
        desc = None
        tpl = self.root.app.templateMgr.find_template('project.description', where=[self], user=user)
        if tpl:
            desc = tpl.render(gws.TemplateRenderInput(args={'project': self}, user=user))

        printers = [p for p in self.printers if user.can_use(p)]
        printers.extend(p for p in self.root.app.printers if user.can_use(p))
        if not printers:
            printers = [self.root.app.defaultPrinter]

        return gws.Props(
            actions=self.root.app.actionMgr.actions_for_project(self, user),
            client=self.client or self.root.app.client,
            description=desc.content if desc else '',
            map=self.map,
            metadata=gws.base.metadata.props(self.metadata),
            exporters=self.root.app.exporterMgr.list_exporters([self, self.root.app], user=user),
            models=[],
            overviewMap=self.overviewMap,
            printers=printers,
            title=self.title,
            uid=self.uid,
        )

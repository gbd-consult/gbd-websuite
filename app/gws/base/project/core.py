import gws
import gws.base.action
import gws.base.client
import gws.base.map
import gws.base.printer
import gws.base.template
import gws.base.web
import gws.lib.metadata
import gws.types as t

_DEFAULT_TEMPLATES = [
    gws.Config(
        uid='gws.base.project.templates.project_description',
        type='html',
        path=gws.dirname(__file__) + '/templates/project_description.cx.html',
        subject='project.description',
        access=gws.PUBLIC,
    ),
]


@gws.ext.config.project('default')
class Config(gws.ConfigWithAccess):
    """Project configuration"""

    api: t.Optional[gws.base.action.manager.Config]  #: project-specific actions
    assets: t.Optional[gws.base.web.site.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.base.client.Config]  #: project-specific gws client configuration
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.base.map.Config]  #: Map configuration
    metadata: t.Optional[gws.Metadata]  #: project metadata
    overviewMap: t.Optional[gws.base.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.base.printer.Config]  #: print configuration
    templates: t.Optional[t.List[gws.ext.config.template]]  #: project info templates
    title: str = ''  #: project title


@gws.ext.props.project('default')
class Props(gws.Props):
    actions: t.List[gws.ext.props.action]
    client: t.Optional[gws.base.client.Props]
    description: str
    locales: t.List[str]
    map: gws.ext.props.map
    metadata: gws.lib.metadata.Props
    overviewMap: gws.ext.props.map
    printer: gws.base.printer.Props
    title: str
    uid: str


@gws.ext.object.project('default')
class Object(gws.Node, gws.IProject):
    overview_map: gws.base.map.Object
    printer: gws.base.printer.Object
    title: str

    def configure(self):
        self.uid = self.var('uid')

        self.metadata = gws.lib.metadata.from_config(self.var('metadata'))
        gws.lib.metadata.extend(self.metadata, self.root.app.metadata)

        # title at the top level config preferred
        title = self.var('title') or self.metadata.get('title') or self.var('uid')
        self.metadata.title = title
        self.title = title

        gws.log.info(f'configuring project {self.uid!r}')

        self.actionMgr = self.create_child_if_configured(gws.base.action.manager.Object, self.var('api'))

        p = self.var('assets')
        self.assetsRoot = gws.WebDocumentRoot(p) if p else None

        self.localeUids = self.var('locales') or self.root.app.localeUids

        self.map = self.create_child_if_configured(gws.ext.object.map, self.var('map'))
        self.printer = self.create_child_if_configured(gws.base.printer.Object, self.var('printer'))
        #
        # self.overview_map = self.root.create_optional(gws.base.map.Object, self.var('overviewMap'))
        #
        self.templateMgr = self.create_child(gws.base.template.manager.Object, gws.Config(
            templates=self.var('templates'),
            defaults=_DEFAULT_TEMPLATES))

        self.client = self.create_child_if_configured(gws.base.client.Object, self.var('client'))

    def props(self, user):
        desc = self.templateMgr.render(
            gws.TemplateRenderInput(args={'project': self, 'user': user}),
            subject='project.description'
        )

        return gws.Props(
            actions=self.root.app.actionMgr.actions_for(user, self.actionMgr),
            client=self.client or self.root.app.client,
            description=desc.content if desc else '',
            map=self.map,
            metadata=gws.lib.metadata.props(self.metadata),
            printer=self.printer,
            title=self.title,
            uid=self.uid,
        )

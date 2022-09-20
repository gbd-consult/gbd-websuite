import gws
import gws.base.action
import gws.base.client
import gws.base.map
import gws.base.printer
import gws.base.template
import gws.base.web
import gws.base.metadata
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
    metadata: t.Optional[gws.base.metadata.Config]  #: project metadata
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
    metadata: gws.base.metadata.Props
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
        self.metadata = gws.base.metadata.from_config(self.var('metadata')).extend(self.root.app.metadata)

        # title at the top level config preferred
        title = self.var('title') or self.metadata.get('title') or self.var('uid')
        self.metadata.set('title', title)
        self.title = title

        gws.log.info(f'configuring project {self.uid!r}')

        self.actionMgr = self.create_child(gws.base.action.manager.Object, self.var('api'), optional=True)

        p = self.var('assets')
        self.assetsRoot = gws.WebDocumentRoot(p) if p else None

        self.localeUids = self.var('locales') or self.root.app.localeUids

        self.map = self.create_child(gws.ext.object.map, self.var('map'), optional=True)
        self.printer = self.create_child(gws.base.printer.Object, self.var('printer'), optional=True)
        #
        # self.overview_map = self.root.create_optional(gws.base.map.Object, self.var('overviewMap'))
        #
        self.templateMgr = self.create_child(gws.base.template.manager.Object, gws.Config(
            templates=self.var('templates'),
            defaults=_DEFAULT_TEMPLATES))

        self.client = self.create_child(gws.base.client.Object, self.var('client'), optional=True)

    def props(self, user):
        p = gws.Data()

        desc = self.render_description()
        p.description = desc.content if desc else ''

        p.actions = self.root.app.actionMgr.actions_for(user, self.actionMgr)
        p.client = self.client or self.root.app.client
        p.map = self.map
        p.metadata = self.metadata
        # p.overviewMap=self.overview_map
        p.printer = self.printer
        p.title = self.title
        p.uid = self.uid

        return p

    def render_description(self, args=None):
        tpl = self.templateMgr.find(subject='project.description')
        if not tpl:
            return
        args = gws.merge({
            'project': self,
            'meta': self.metadata.values
        }, args)
        return tpl.render(gws.TemplateRenderInput(args=args))

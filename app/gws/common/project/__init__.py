import gws.common.client
import gws.common.map
import gws.common.printer
import gws.common.search
import gws.common.template
import gws.config
import gws.gis
import gws.types as t


class Config(t.Config):
    """project configuration"""

    access: t.Optional[t.Access]  #: access rights
    actions: t.Optional[t.List[t.ext.action.Config]]  #: project-specific actions
    assets: t.Optional[t.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.common.client.Config]  #: project-specific gws client configuration
    description: t.Optional[t.TemplateConfig]  #: template for the project description
    locale: t.Optional[str]  #: Project locale
    map: t.Optional[gws.common.map.Config]  #: Map configuration
    meta: t.Optional[t.MetaConfig]  #: project metadata
    multiMatch: t.Optional[t.regex]  #: filename pattern for a multi-project template
    overviewMap: t.Optional[gws.common.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.common.printer.Config]  #: printer configuration
    search: t.Optional[gws.common.search.Config]  #: project-wide search configuration
    title: str = ''  #: project title
    uid: t.Optional[str]  #: unique id


class Props(t.Data):
    client: gws.common.client.Props
    description: str = ''
    locale: str
    map: gws.common.map.Props
    meta: t.MetaData
    overviewMap: gws.common.map.Props = None
    printer: gws.common.printer.Props = None
    title: str
    uid: str


class Object(gws.PublicObject, t.ProjectObject):
    def __init__(self):
        super().__init__()

        self.client: gws.common.client.Object = None
        self.locale = ''
        self.map: gws.common.map.Object = None
        self.overview_map: gws.common.map.Object = None
        self.printer: gws.common.printer.Object = None
        self.description_template: t.TemplateObject = None
        self.title = ''
        self.meta: t.MetaData = {}

    def configure(self):
        super().configure()

        self.meta = self.configure_meta()
        self.uid = self.var('uid') or gws.as_uid(self.title)

        self.locale = self.var('locale', parent=True)

        p = self.var('map')
        if p:
            self.map = self.add_child(gws.common.map.Object, p)

        p = self.var('overviewMap')
        if p:
            p.uid = 'overview'
            self.overview_map = self.add_child(gws.common.map.Object, p)

        p = self.var('printer')
        if p:
            self.printer = self.add_child(gws.common.printer.Object, p)

        p = self.var('client')
        if p:
            self.client = self.add_child(gws.common.client.Object, p)

        self.description_template = self.add_child(
            'gws.ext.template',
            self.var('description') or gws.common.template.builtin_config('project_description')
        )

        for p in self.var('actions', []):
            a = self.add_child('gws.ext.action', p)
            # project-specific actions must have the project id, see web.actions
            a.uid = self.uid + '.' + p.type

        for p in self.var('search.providers', default=[]):
            self.add_child('gws.ext.search.provider', p)

    def configure_meta(self):
        # @TODO merge with layer
        m = self.var('meta') or t.MetaData()
        # title at the top level config preferred
        if self.var('title'):
            m.title = self.var('title')
        self.title = m.title
        return m

    def description(self, options=None):
        ctx = gws.defaults(options, {
            'project': self,
            'meta': self.meta,
        })
        return self.description_template.render(ctx).content

    @property
    def props(self):
        cc = self.client or getattr(self.parent, 'client', None)

        return gws.compact({
            'client': cc,
            'description': self.description(),
            'map': self.map,
            'meta': self.meta,
            'overviewMap': self.overview_map,
            'printer': self.printer,
            'title': self.title,
            'uid': self.uid,
        })

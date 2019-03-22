import gws.common.api
import gws.common.client
import gws.common.map
import gws.common.printer
import gws.common.search
import gws.common.template
import gws.web.site

import gws.types as t


class ApiConfig(t.Config):
    """Project-specific server actions"""

    access: t.Optional[t.Access]  #: default access mode
    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


class Config(t.Config):
    """Project configuration"""

    access: t.Optional[t.Access]  #: access rights
    api: t.Optional[gws.common.api.Config]  #: project-specific actions
    assets: t.Optional[t.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.common.client.Config]  #: project-specific gws client configuration
    description: t.Optional[t.TemplateConfig]  #: template for the project description
    locale: t.Optional[str]  #: Project locale
    map: t.Optional[gws.common.map.Config]  #: Map configuration
    meta: t.Optional[t.MetaConfig] = {}  #: project metadata
    multi: t.Optional[t.regex]  #: filename pattern for a multi-project template
    overviewMap: t.Optional[gws.common.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.common.printer.Config]  #: printer configuration
    search: t.Optional[gws.common.search.Config] = {}  #: project-wide search configuration
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

        self.api: gws.common.api.Object = None
        self.assets_root: t.DocumentRootConfig = None
        self.client: gws.common.client.Object = None
        self.description_template: t.TemplateObject = None
        self.locale = ''
        self.map: gws.common.map.Object = None
        self.meta: t.MetaData = {}
        self.overview_map: gws.common.map.Object = None
        self.printer: gws.common.printer.Object = None
        self.title = ''

    def configure(self):
        super().configure()

        self.meta = self.var('meta')
        # title at the top level config preferred
        if self.var('title'):
            self.meta.title = self.var('title')
        self.title = self.meta.title

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

        self.description_template = self.add_child(
            'gws.ext.template',
            self.var('description') or gws.common.template.builtin_config('project_description')
        )

        search = self.var('search')
        if search.enabled and search.providers:
            for p in search.providers:
                self.add_child('gws.ext.search.provider', p)

        p = self.var('api')
        self.api = self.add_child(gws.common.api.Object, p) if p else None

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
            self.client = self.add_child(gws.common.client.Object, p)

        self.assets_root = self.var('assets')

    @property
    def description(self):
        ctx = {
            'project': self,
            'meta': self.meta,
        }
        return self.description_template.render(ctx).content

    @property
    def props(self):
        cc = self.client or getattr(self.parent, 'client', None)

        return gws.compact({
            'client': cc,
            'description': self.description,
            'map': self.map,
            'meta': self.meta,
            'overviewMap': self.overview_map,
            'printer': self.printer,
            'title': self.title,
            'uid': self.uid,
        })

import gws.common.api
import gws.common.client
import gws.common.map
import gws.common.printer
import gws.common.search
import gws.common.template
import gws.common.metadata
import gws.web.site

import gws.types as t


class ApiConfig(t.WithAccess):
    """Project-specific server actions"""

    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


class Config(t.WithAccess):
    """Project configuration"""

    api: t.Optional[gws.common.api.Config]  #: project-specific actions
    assets: t.Optional[gws.web.site.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.common.client.Config]  #: project-specific gws client configuration
    description: t.Optional[t.ext.template.Config]  #: template for the project description
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.common.map.Config]  #: Map configuration
    meta: t.Optional[gws.common.metadata.Config] = {}  #: project metadata
    multi: t.Optional[t.Regex]  #: filename pattern for a multi-project template
    overviewMap: t.Optional[gws.common.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.common.printer.Config]  #: printer configuration
    search: t.Optional[gws.common.search.Config] = {}  #: project-wide search configuration
    title: str = ''  #: project title
    uid: t.Optional[str]  #: unique id


class Props(t.Data):
    actions: t.List[t.ext.action.Props]
    client: t.Optional[gws.common.client.Props]
    description: str
    locales: t.List[str]
    map: gws.common.map.Props
    meta: t.MetaData
    overviewMap: gws.common.map.Props
    printer: gws.common.printer.Props
    title: str
    uid: str


#:export IProject
class Object(gws.Object, t.IProject):
    def __init__(self):
        super().__init__()

        self.api: t.IApi = None
        self.assets_root: t.DocumentRoot = None
        self.client: t.IClient = None
        self.description_template: t.ITemplate = None
        self.locales = []
        self.map: t.IMap = None
        self.meta: t.MetaData = {}
        self.overview_map: t.IMap = None
        self.printer: t.IPrinter = None
        self.title = ''

    def configure(self):
        super().configure()

        self.meta = gws.common.metadata.read(self.var('meta'))
        # title at the top level config preferred
        if self.var('title'):
            self.meta.title = self.var('title')

        self.title = self.meta.title

        self.locales = self.var('locales', parent=True, default=['en_CA'])
        self.assets_root = gws.web.site.document_root(self.var('assets'))

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

        p = self.var('description')
        self.description_template = self.add_child('gws.ext.template', p or gws.common.template.builtin_config('project_description'))

        p = self.var('search')
        if p and p.enabled and p.providers:
            for s in p.providers:
                self.add_child('gws.ext.search.provider', s)

        p = self.var('api')
        self.api = self.add_child(gws.common.api.Object, p) if p else None

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
            self.client = self.add_child(gws.common.client.Object, p)

    @property
    def description(self):
        ctx = {
            'project': self,
            'meta': self.meta,
        }
        return self.description_template.render(ctx).content

    @property
    def props(self):
        return Props(
            actions=gws.merge(
                {},
                self.root.application.api.actions,
                self.api.actions if self.api else {}),
            client=self.client or getattr(self.parent, 'client', None),
            description=self.description,
            map=self.map,
            meta=self.meta,
            overviewMap=self.overview_map,
            printer=self.printer,
            title=self.title,
            uid=self.uid,
        )

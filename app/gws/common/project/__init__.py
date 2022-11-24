import gws.common.api
import gws.common.client
import gws.common.map
import gws.common.metadata
import gws.common.printer
import gws.common.search
import gws.common.search.provider
import gws.common.template
import gws.gis.extent
import gws.gis.proj
import gws.tools.units
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
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.common.map.Config]  #: Map configuration
    meta: t.Optional[gws.common.metadata.Config] = {}  #: project metadata
    overviewMap: t.Optional[gws.common.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.common.printer.Config]  #: printer configuration
    search: t.Optional[gws.common.search.Config] = {}  #: project-wide search configuration
    templates: t.Optional[t.List[t.ext.template.Config]]  #: project info templates
    title: str = ''  #: project title
    uid: t.Optional[str]  #: unique id
    qgisfilter: t.Optional[str]  #: dynamic qgis filter


class Props(t.Data):
    actions: t.List[t.ext.action.Props]
    client: t.Optional[gws.common.client.Props]
    description: str
    locales: t.List[str]
    map: gws.common.map.Props
    meta: gws.common.metadata.Props
    overviewMap: gws.common.map.Props
    printer: gws.common.printer.Props
    searchCategories: t.List[str]
    title: str
    uid: str


#:export IProject
class Object(gws.Object, t.IProject):
    def configure(self):
        super().configure()

        self.meta: t.MetaData = gws.common.metadata.from_config(self.var('meta'))
        gws.log.info(f'configuring project {self.uid!r}')

        # title at the top level config preferred
        title = self.var('title') or self.meta.title or self.uid
        self.meta.title = title
        self.title: str = title

        self.locale_uids: t.List[str] = self.var('locales', parent=True, default=['en_CA'])
        self.assets_root: t.Optional[t.DocumentRoot] = gws.web.site.document_root(self.var('assets'))

        p = self.var('map')
        self.map: t.Optional[t.IMap] = self.create_child(gws.common.map.Object, p) if p else None

        p = self.var('overviewMap')
        self.overview_map: t.Optional[t.IMap] = self.create_child(gws.common.map.Object, p) if p else None
        if self.overview_map:
            self.overview_map.uid = 'overview'

        p = self.var('printer')
        self.printer: t.Optional[t.IPrinter] = self.create_child(gws.common.printer.Object, p) if p else None

        self.templates: t.List[t.ITemplate] = gws.common.template.bundle(self, self.var('templates'), gws.common.template.BUILTINS)

        self.search_providers = []

        p = self.var('search')
        if p and p.enabled and p.providers:
            for s in p.providers:
                self.search_providers.append(self.create_child('gws.ext.search.provider', s))

        p = self.var('api')
        self.api: t.Optional[t.IApi] = self.create_child(gws.common.api.Object, p) if p else None

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
        self.client: t.Optional[t.IClient] = self.create_child(gws.common.client.Object, p) if p else None

        self.qgisfilter = self.var('qgisfilter')


    @property
    def description(self):
        context = {'project': self, 'meta': self.meta}
        tpl = gws.common.template.find(self.templates, subject='project.description')
        return tpl.render(context).content

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
            meta=gws.common.metadata.props(self.meta),
            overviewMap=self.overview_map,
            printer=self.printer,
            searchCategories=sorted(set(c for p in self.search_providers for c in p.categories)),
            title=self.title,
            uid=self.uid,
        )

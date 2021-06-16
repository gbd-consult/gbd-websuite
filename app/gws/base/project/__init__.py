import gws.base.api
import gws.base.client
import gws.base.map
import gws.base.metadata
import gws.base.printer
import gws.base.search
import gws.base.template
import gws.gis.extent
import gws.gis.proj
import gws.lib.units
import gws.web.site

import gws.types as t


class ApiConfig(t.WithAccess):
    """Project-specific server actions"""

    actions: t.Optional[t.List[t.ext.action.Config]]  #: available actions


class Config(t.WithAccess):
    """Project configuration"""

    api: t.Optional[gws.base.api.Config]  #: project-specific actions
    assets: t.Optional[gws.web.site.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.base.client.Config]  #: project-specific gws client configuration
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.base.map.Config]  #: Map configuration
    meta: t.Optional[gws.base.metadata.Config] = {}  #: project metadata
    overviewMap: t.Optional[gws.base.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.base.printer.Config]  #: printer configuration
    search: t.Optional[gws.base.search.Config] = {}  #: project-wide search configuration
    templates: t.Optional[t.List[t.ext.template.Config]]  #: project info templates
    title: str = ''  #: project title
    uid: t.Optional[str]  #: unique id


class Props(t.Data):
    actions: t.List[t.ext.action.Props]
    client: t.Optional[gws.base.client.Props]
    description: str
    locales: t.List[str]
    map: gws.base.map.Props
    meta: gws.base.metadata.Props
    overviewMap: gws.base.map.Props
    printer: gws.base.printer.Props
    title: str
    uid: str


#:export IProject
class Object(gws.Object, t.IProject):
    def configure(self):
        super().configure()

        self.meta: t.MetaData = gws.base.metadata.from_config(self.var('meta'))
        gws.log.info(f'configuring project {self.uid!r}')

        # title at the top level config preferred
        title = self.var('title') or self.meta.title or self.uid
        self.meta.title = title
        self.title: str = title

        self.locale_uids: t.List[str] = self.var('locales', parent=True, default=['en_CA'])
        self.assets_root: t.Optional[t.DocumentRoot] = gws.web.site.document_root(self.var('assets'))

        p = self.var('map')
        self.map: t.Optional[t.IMap] = self.create_child(gws.base.map.Object, p) if p else None

        p = self.var('overviewMap')
        self.overview_map: t.Optional[t.IMap] = self.create_child(gws.base.map.Object, p) if p else None
        if self.overview_map:
            self.overview_map.uid = 'overview'

        p = self.var('printer')
        self.printer: t.Optional[t.IPrinter] = self.create_child(gws.base.printer.Object, p) if p else None

        self.templates: t.List[t.ITemplate] = gws.base.template.bundle(self, self.var('templates'), gws.base.template.BUILTINS)

        p = self.var('search')
        if p and p.enabled and p.providers:
            for s in p.providers:
                self.create_child('gws.ext.search.provider', s)

        p = self.var('api')
        self.api: t.Optional[t.IApi] = self.create_child(gws.base.api.Object, p) if p else None

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
        self.client: t.Optional[t.IClient] = self.create_child(gws.base.client.Object, p) if p else None

    @property
    def description(self):
        context = {'project': self, 'meta': self.meta}
        tpl = gws.base.template.find(self.templates, subject='project.description')
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
            meta=gws.base.metadata.props(self.meta),
            overviewMap=self.overview_map,
            printer=self.printer,
            title=self.title,
            uid=self.uid,
        )

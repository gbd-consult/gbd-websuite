import gws
import gws.base.api
import gws.base.api
import gws.base.auth
import gws.base.client
import gws.base.map
import gws.base.metadata
import gws.base.printer
import gws.base.search
import gws.base.template
import gws.base.web
import gws.lib.extent
import gws.lib.intl
import gws.lib.proj
import gws.lib.units
import gws.types as t


class Config(gws.WithAccess):
    """Project configuration"""

    api: t.Optional[gws.base.api.Config]  #: project-specific actions
    assets: t.Optional[gws.base.web.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.base.client.Config]  #: project-specific gws client configuration
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.base.map.Config]  #: Map configuration
    metaData: t.Optional[gws.base.metadata.Config]  #: project metadata
    overviewMap: t.Optional[gws.base.map.Config]  #: Overview map configuration
    printer: t.Optional[gws.base.printer.Config]  #: print configuration
    search: t.Optional[gws.base.search.Config] = {}  # type: ignore #: project-wide search configuration
    templates: t.Optional[t.List[gws.ext.template.Config]]  #: project info templates
    title: str = ''  #: project title


class Props(gws.Props):
    actions: t.List[gws.ext.action.Props]
    client: t.Optional[gws.base.client.Props]
    description: str
    locales: t.List[str]
    map: gws.base.map.Props
    metaData: gws.base.metadata.Props
    overviewMap: gws.base.map.Props
    printer: gws.base.printer.Props
    title: str
    uid: str


class Object(gws.Object):
    api: gws.base.api.Object
    assets_root: t.Optional[gws.DocumentRoot]
    client: t.Optional[gws.base.client.Object]
    locale_uids: t.List[str]
    map: gws.base.map.Object
    metadata: gws.base.metadata.Object
    overview_map: gws.base.map.Object
    printer: gws.base.printer.Object
    search_providers: t.List[gws.ISearchProvider]
    templates: gws.base.template.bundle.Object
    title: str

    def configure(self):
        self.metadata = self.create_child(
            gws.base.metadata.Object,
            self.var('metaData', with_parent=True) or gws.base.metadata.Config(title=self.var('title')))

        # title at the top level config preferred
        title = self.var('title') or self.metadata.get('title') or self.var('uid')
        self.metadata.set('title', title)
        self.title = title

        self.set_uid(self.var('uid') or gws.as_uid(self.title))

        gws.log.info(f'configuring project {self.uid!r}')

        self.api = self.create_child_if_config(gws.base.api.Object, self.var('api'))
        self.assets_root = gws.base.web.document_root_from_config(self.var('assets'))
        self.locale_uids = self.var('locales', with_parent=True, default=['en_CA'])
        self.map = self.create_child_if_config(gws.base.map.Object, self.var('map'))
        self.printer = self.create_child_if_config(gws.base.printer.Object, self.var('printer'))

        self.overview_map = self.create_child_if_config(gws.base.map.Object, self.var('overviewMap'))
        if self.overview_map:
            self.overview_map.set_uid(self.uid + '.overview')

        self.templates = gws.base.template.bundle.create(self.root, gws.Config(templates=self.var('templates'), withBuiltins=True))

        self.search_providers = []

        p = self.var('search')
        if p and p.enabled and p.providers:
            for s in p.providers:
                self.search_providers.append(
                    t.cast('gws.ISearchProvider', self.create_child('gws.ext.search.provider', s)))

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
        self.client = self.create_child_if_config(gws.base.client.Object, p)

    @property
    def description(self):
        context = {'project': self, 'meta': self.metadata.record}
        tpl = self.templates.find(subject='project.description')
        return tpl.render(context).content if tpl else ''

    @property
    def props(self):
        actions = gws.merge(
            gws.get(self.parent, 'api.actions'),
            gws.get(self, 'api.actions'))
        return Props(
            actions=list(actions.values()),
            client=self.client or getattr(self.parent, 'client', None),
            description=self.description,
            map=self.map,
            metaData=self.metadata,
            overviewMap=self.overview_map,
            printer=self.printer,
            title=self.title,
            uid=self.uid,
        )

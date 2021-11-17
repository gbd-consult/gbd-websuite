import gws
import gws.base.api
import gws.base.client
import gws.base.map
import gws.base.printer
import gws.base.search
import gws.base.template
import gws.base.web
import gws.lib.metadata
import gws.types as t

_DEFAULT_TEMPLATES = [
    gws.Config(
        type='html',
        path=gws.dirname(__file__) + '/templates/project_description.cx.html',
        subject='project.description',
        access=[{'role': 'all', 'type': 'allow'}],
    ),
]


class Config(gws.WithAccess):
    """Project configuration"""

    api: t.Optional[gws.base.api.Config]  #: project-specific actions
    assets: t.Optional[gws.base.web.DocumentRootConfig]  #: project-specific assets options
    client: t.Optional[gws.base.client.Config]  #: project-specific gws client configuration
    locales: t.Optional[t.List[str]]  #: project locales
    map: t.Optional[gws.base.map.Config]  #: Map configuration
    metadata: t.Optional[gws.lib.metadata.Config]  #: project metadata
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
    metadata: gws.lib.metadata.Props
    overviewMap: gws.base.map.Props
    printer: gws.base.printer.Props
    title: str
    uid: str


class Object(gws.Node, gws.IProject):
    overview_map: gws.base.map.Object
    printer: gws.base.printer.Object

    def configure(self):
        self.metadata = gws.lib.metadata.from_config(self.var('metadata')).extend(self.root.application.metadata)

        # title at the top level config preferred
        title = self.var('title') or self.metadata.get('title') or self.var('uid')
        self.metadata.set('title', title)
        self.title = title

        self.set_uid(self.var('uid') or gws.to_uid(self.title))

        gws.log.info(f'configuring project {self.uid!r}')

        self.api = self.create_child_if_config(gws.base.api.Object, self.var('api'))
        self.assets_root = gws.base.web.create_document_root(self.var('assets'))
        self.locale_uids = self.var('locales', with_parent=True, default=['en_CA'])
        self.map = self.create_child_if_config(gws.base.map.Object, self.var('map'))
        self.printer = self.create_child_if_config(gws.base.printer.Object, self.var('printer'))

        self.overview_map = self.create_child_if_config(gws.base.map.Object, self.var('overviewMap'))
        if self.overview_map:
            self.overview_map.set_uid(self.uid + '.overview')

        self.templates = gws.base.template.bundle.create(
            self.root,
            items=self.var('templates'),
            defaults=_DEFAULT_TEMPLATES,
            parent=self)

        self.search_providers = []
        p = self.var('search')
        if p and p.enabled and p.providers:
            self.search_providers = self.create_children('gws.ext.search.provider', p.providers)

        p = self.var('client')
        if p:
            self.client = self.create_child(
                gws.base.client.Object,
                gws.merge(p, parentClient=self.parent.var('client')))

    @property
    def description(self):
        tpl = self.templates.find(subject='project.description')
        if not tpl:
            return ''
        context = {
            'project': self,
            'meta': self.metadata.values
        }
        return tpl.render(gws.TemplateRenderInput(context=context)).content

    def props_for(self, user):
        app_api = self.root.application.api
        actions = self.api.actions_for(user, app_api) if self.api else app_api.actions_for(user)

        return gws.Data(
            actions=list(actions.values()),
            client=self.client or self.root.application.client,
            description=self.description,
            map=self.map,
            metadata=self.metadata,
            overviewMap=self.overview_map,
            printer=self.printer,
            title=self.title,
            uid=self.uid,
        )

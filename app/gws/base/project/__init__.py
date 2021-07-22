import gws
import gws.base.api.action
import gws.base.api.action
import gws.base.auth
import gws.base.client
import gws.base.map
import gws.base.metadata
import gws.base.print
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
    print: t.Optional[gws.base.print.Config]  #: print configuration
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
    print: gws.base.print.Props
    title: str
    uid: str


class Object(gws.Node):
    api: gws.base.api.Object
    assets_root: t.Optional[gws.base.web.DocumentRoot]
    client: t.Optional[gws.base.client.Object]
    locale_uids: t.List[str]
    map: gws.base.map.Object
    metadata: gws.base.metadata.Object
    overview_map: gws.base.map.Object
    print: gws.base.print.Object
    templates: gws.base.template.Bundle
    title: str

    def configure(self):
        p = self.var('metaData')
        self.metadata = self.create_child(gws.base.metadata.Object, p)

        # title at the top level config preferred
        title = self.var('title') or self.metadata.get('title') or self.uid
        self.metadata.set('title', title)
        self.title: str = title

        self.set_uid(self.title)

        gws.log.info(f'configuring project {self.uid!r}')

        p = self.var('api')
        self.api = self.create_child(gws.base.api.Object, p) if p else None

        self.locale_uids = self.var('locales', with_parent=True, default=['en_CA'])

        p = self.var('assets')
        self.assets_root = self.create_child(gws.base.web.DocumentRoot, p) if p else None

        p = self.var('map')
        self.map = self.create_child(gws.base.map.Object, p) if p else None

        p = self.var('overviewMap')
        self.overview_map = self.create_child(gws.base.map.Object, p) if p else None
        if self.overview_map:
            self.overview_map.set_uid(self.uid + '.overview')

        p = self.var('print')
        self.print = self.create_child(gws.base.print.Object, p) if p else None

        p = self.var('templates')
        self.templates = t.cast(
            gws.base.template.Bundle,
            self.create_child(
                gws.base.template.Bundle,
                gws.Config(templates=p, defaults=gws.base.template.BUILTINS)))
        #
        # p = self.var('search')
        # if p and p.enabled and p.providers:
        #     for s in p.providers:
        #         self.create_child('gws.ext.search.provider', s)

        p = self.var('client')
        if p:
            p.parentClient = self.parent.var('client')
        self.client = self.create_child(gws.base.client.Object, p) if p else None

    @property
    def description(self):
        context = {'project': self, 'meta': self.metadata.values}
        tpl = self.templates.find(subject='project.description')
        return tpl.render(context).content if tpl else ''

    @property
    def props(self):
        actions = gws.merge(
            {},
            gws.get(self.parent, 'api.actions'),
            gws.get(self, 'api.actions'))
        return Props(
            actions=list(actions.values()),
            client=self.client or getattr(self.parent, 'client', None),
            description=self.description,
            map=self.map,
            metaData=self.metadata,
            overviewMap=self.overview_map,
            print=self.print,
            title=self.title,
            uid=self.uid,
        )


class InfoResponse(gws.Response):
    project: Props
    locale: gws.lib.intl.Locale
    user: t.Optional[gws.base.auth.UserProps]


@gws.ext.Object('action.project')
class Action(gws.base.api.action.Object):
    """Project information action"""

    @gws.ext.command('api.project.info')
    def info(self, req: gws.IWebRequest, p: gws.Params) -> InfoResponse:
        """Return the project configuration"""

        project = req.require_project(p.projectUid)

        locale_uid = p.localeUid
        if locale_uid not in project.locale_uids:
            locale_uid = project.locale_uids[0]

        return InfoResponse(
            project=project.props_for(req.user),
            locale=gws.lib.intl.locale(locale_uid),
            user=None if req.user.is_guest else req.user.props)

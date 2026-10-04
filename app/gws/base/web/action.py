"""Web action for pages, assets and files."""

from typing import Optional, cast


import gws
import gws.base.action
import gws.base.client.bundles
import gws.lib.mime
import gws.lib.osx
import gws.lib.intl


class TemplateArgs(gws.TemplateArgs):
    """Asset template arguments."""

    project: Optional[gws.Project]
    """Current project."""
    projects: list[gws.Project]
    """List of user projects."""
    req: gws.WebRequester
    """Requester object."""
    user: gws.User
    """Current user."""
    params: dict
    """Request parameters."""
    locale: gws.Locale
    """Locale object."""


@gws.ext.config.action('web')
class Config(gws.base.action.Config):
    """Serves web pages, assets and files to the browser."""

    pass


@gws.ext.props.action('web')
class Props(gws.base.action.Props):
    pass


class AssetRequest(gws.Request):
    """Asset request."""

    path: str
    """Asset path, relative to the assets directory."""


class PageRequest(gws.Request):
    """Page request."""

    name: str
    """Page name, ``home`` or ``project``."""


class AssetResponse(gws.Request):
    """Asset response for API requests."""

    content: str
    """Asset content."""
    mimeType: str
    """Asset MIME type."""


class FileRequest(gws.Request):
    """Request for a file stored in a feature field."""

    preview: bool = False
    """Return a preview instead of the file."""
    modelUid: str
    """Uid of the model."""
    fieldName: str
    """Name of the file field."""
    featureUid: str
    """Uid of the feature."""


@gws.ext.object.action('web')
class Object(gws.base.action.Object):
    """Web action, serves pages, assets and files."""

    @gws.ext.command.api('webAsset')
    def api_asset(self, req: gws.WebRequester, p: AssetRequest) -> AssetResponse:
        """Return an asset of the site or of the project."""
        res = self._serve_path(req, p)
        if res.contentPath:
            res.content = gws.u.read_file_b(res.contentPath)
        return AssetResponse(content=res.content, mimeType=res.mimeType)

    @gws.ext.command.get('webAsset')
    def http_asset(self, req: gws.WebRequester, p: AssetRequest) -> gws.ContentResponse:
        """Serve an asset of the site or of the project."""
        res = self._serve_path(req, p)
        return res

    @gws.ext.command.get('webPage')
    def get_page(self, req: gws.WebRequester, p: PageRequest) -> gws.ContentResponse:
        """Render the application home page or a project page."""
        tpl = None
        project = None
        
        if p.name == 'home':
            tpl = self.root.app.templateMgr.find_template('application.home', where=[], user=req.user)
        if p.name == 'project':
            project = req.user.require_project(p.projectUid)
            tpl = self.root.app.templateMgr.find_template('project.home', where=[project], user=req.user)
        if not tpl:
            raise gws.NotFoundError('template not found for {p.name=}')
        
        return self._serve_template(req, p, tpl, project=project)

    @gws.ext.command.get('webDownload')
    def download(self, req: gws.WebRequester, p) -> gws.ContentResponse:
        """Serve an asset as a download attachment."""
        res = self._serve_path(req, p)
        if res.contentPath:
            res.contentFilename = gws.lib.osx.parse_path(res.contentPath).filename
        return res

    @gws.ext.command.get('webFile')
    def file(self, req: gws.WebRequester, p: FileRequest) -> gws.ContentResponse:
        """Serve a file stored in a feature field."""
        model = cast(gws.Model, req.user.require(p.modelUid, gws.ext.object.model, gws.Access.read))
        field = model.field(p.fieldName)
        if not field:
            raise gws.NotFoundError()
        fn = getattr(field, 'handle_web_file_request', None)
        if not fn:
            raise gws.NotFoundError()
        mc = gws.ModelContext(
            op=gws.ModelOperation.read,
            user=req.user,
            project=req.user.require_project(p.projectUid),
            maxDepth=0,
        )
        res = fn(p.featureUid, p.preview, mc)
        if not res:
            raise gws.NotFoundError()
        return res

    @gws.ext.command.get('webSystemAsset')
    def sys_asset(self, req: gws.WebRequester, p: AssetRequest) -> gws.ContentResponse:
        """Serve a client script or style sheet."""
        locale = gws.lib.intl.locale(p.localeUid, self.root.app.localeUids)
        app_templates = f'{gws.c.APP_DIR}/gws/base/application/templates/'

        # only accept '8.0.0.vendor.js' etc or simply 'vendor.js'
        path = p.path
        if path.startswith(self.root.app.version):
            path = path[len(self.root.app.version) + 1 :]

        if path == 'vendor.js':
            return gws.ContentResponse(mimeType=gws.lib.mime.JS, content=gws.base.client.bundles.javascript(self.root, 'vendor', locale))

        if path == 'home.js':
            return gws.ContentResponse(mimeType=gws.lib.mime.JS, content=gws.u.read_file(f'{app_templates}/home.js'))
        # deprecated
        if path == 'util.js':
            return gws.ContentResponse(mimeType=gws.lib.mime.JS, content=gws.u.read_file(f'{app_templates}/home.js'))

        if path == 'home.css':
            return gws.ContentResponse(mimeType=gws.lib.mime.CSS, content=gws.u.read_file(f'{app_templates}/home.css'))

        if path == 'app.js':
            return gws.ContentResponse(mimeType=gws.lib.mime.JS, content=gws.base.client.bundles.javascript(self.root, 'app', locale))

        if path.endswith('.css'):
            s = path.split('.')
            if len(s) != 2:
                raise gws.NotFoundError(f'invalid css request: {p.path=}')
            content = gws.base.client.bundles.css(self.root, 'app', s[0])
            if not content:
                raise gws.NotFoundError(f'invalid css request: {p.path=}')
            return gws.ContentResponse(mimeType=gws.lib.mime.CSS, content=content)

        raise gws.NotFoundError(f'invalid system asset: {p.path=}')

    def _serve_path(self, req: gws.WebRequester, p: AssetRequest):
        """Locate an asset in the project or site assets directory and serve or render it."""
        req_path = str(p.get('path') or '')
        if not req_path:
            raise gws.NotFoundError('no path provided')

        site_assets = req.site.assetsRoot

        project = None
        project_assets = None

        project_uid = p.get('projectUid')
        if project_uid:
            project = req.user.require_project(project_uid)
            project_assets = project.assetsRoot

        real_path = None
        tpl = None

        if project_assets:
            real_path = gws.lib.osx.abs_web_path(req_path, project_assets.dir)
        if not real_path and site_assets:
            real_path = gws.lib.osx.abs_web_path(req_path, site_assets.dir)

        if real_path:
            tpl = self.root.app.templateMgr.template_from_path(real_path)
        # deprecated: support for index.cx.html and project.cx.html
        elif req_path == 'index.cx.html':
            tpl = self.root.app.templateMgr.find_template('application.home', where=[], user=req.user)
        elif req_path == 'project.cx.html':
            tpl = self.root.app.templateMgr.find_template('project.home', where=[project] if project else [], user=req.user)

        if tpl:
            return self._serve_template(req, p, tpl, project)

        if not real_path:
            raise gws.NotFoundError(f'no real path for {req_path=}')

        mime_type = gws.lib.mime.for_path(real_path)

        if not _valid_mime_type(mime_type, project_assets, site_assets):
            # NB: pretend the file doesn't exist
            raise gws.NotFoundError(f'invalid mime path={real_path!r} mime_type={mime_type!r}')

        gws.log.debug(f'serving {real_path!r} for {req_path!r}')
        return gws.ContentResponse(contentPath=real_path, mimeType=mime_type)

    def _serve_template(self, req: gws.WebRequester, p: gws.Request, tpl: gws.Template, project: Optional[gws.Project]):
        """Render a template with the asset template arguments."""
        locale = gws.lib.intl.locale(p.localeUid, project.localeUids if project else self.root.app.localeUids)
        projects = [p for p in self.root.app.projects if req.user.can_use(p)]
        projects.sort(key=lambda p: p.title.lower())

        args = TemplateArgs(
            project=project,
            projects=projects,
            req=req,
            user=req.user,
            params=req.params(),
        )

        return tpl.render(gws.TemplateRenderInput(args=args, locale=locale))


_DEFAULT_ALLOWED_MIME_TYPES = {
    gws.lib.mime.CSS,
    gws.lib.mime.CSV,
    gws.lib.mime.GEOJSON,
    gws.lib.mime.GIF,
    gws.lib.mime.GML,
    gws.lib.mime.GML3,
    gws.lib.mime.GZIP,
    gws.lib.mime.HTML,
    gws.lib.mime.JPEG,
    gws.lib.mime.JS,
    gws.lib.mime.JSON,
    gws.lib.mime.PDF,
    gws.lib.mime.PNG,
    gws.lib.mime.SVG,
    gws.lib.mime.TTF,
    gws.lib.mime.TXT,
    gws.lib.mime.XML,
    gws.lib.mime.ZIP,
}


def _valid_mime_type(mt, project_assets: Optional[gws.WebDocumentRoot], site_assets: Optional[gws.WebDocumentRoot]):
    """Check a MIME type against the allow and deny lists of the assets directories."""
    if project_assets and project_assets.allowMime:
        return mt in project_assets.allowMime
    if site_assets and site_assets.allowMime:
        return mt in site_assets.allowMime
    if mt not in _DEFAULT_ALLOWED_MIME_TYPES:
        return False
    if project_assets and project_assets.denyMime:
        return mt not in project_assets.denyMime
    if site_assets and site_assets.denyMime:
        return mt not in site_assets.denyMime
    return True

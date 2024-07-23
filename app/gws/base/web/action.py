"""Handle dynamic assets.

An asset is a file located in a global or project-specific ``assets`` directory.

In order to access a project asset, the user must have ``read`` permission for the project itself.

When the Web application receives a ``webAsset`` request with a ``path`` argument, it first checks the project-specific assets directory,
and then the global dir.

If the file is found, and its name matches :obj:`gws.base.template.manager.TEMPLATE_TYPES`, a respective ``Template`` object is generated on the fly and rendered.
The renderer is passed a :obj:`TemplateArgs` object as an argument.
The :obj:`gws.Response` object returned from rendering is passed back to the user.

If the file is not a template and matches the ``allowMime/denyMime`` filter, its content is returned to the user.
"""

from typing import Optional, cast

import os
import re

import gws
import gws.base.action
import gws.base.client.bundles
import gws.base.template
import gws.lib.mime
import gws.lib.osx
import gws.lib.intl

gws.ext.new.action('web')


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


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class AssetRequest(gws.Request):
    path: str


class AssetResponse(gws.Request):
    content: str
    mime: str


class FileRequest(gws.Request):
    preview: bool = False
    modelUid: str
    fieldName: str
    featureUid: str


class Object(gws.base.action.Object):
    """Web action"""

    @gws.ext.command.api('webAsset')
    def api_asset(self, req: gws.WebRequester, p: AssetRequest) -> AssetResponse:
        """Return an asset under the given path and project"""
        r = self._serve_path(req, p)
        if r.contentPath:
            r.content = gws.u.read_file_b(r.contentPath)
        return AssetResponse(content=r.content, mime=r.mime)

    @gws.ext.command.get('webAsset')
    def http_asset(self, req: gws.WebRequester, p: AssetRequest) -> gws.ContentResponse:
        r = self._serve_path(req, p)
        return r

    @gws.ext.command.get('webDownload')
    def download(self, req: gws.WebRequester, p) -> gws.ContentResponse:
        r = self._serve_path(req, p)
        r.asAttachment = True
        return r

    @gws.ext.command.get('webFile')
    def file(self, req: gws.WebRequester, p: FileRequest) -> gws.ContentResponse:
        model = cast(gws.Model, req.user.acquire(p.modelUid, gws.ext.object.model, gws.Access.read))
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
        locale = gws.lib.intl.locale(p.localeUid, self.root.app.localeUids)

        # only accept '8.0.0.vendor.js' etc or simply 'vendor.js'
        path = p.path
        if path.startswith(self.root.app.version):
            path = path[len(self.root.app.version) + 1:]

        if path == 'vendor.js':
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'vendor', locale))

        if path == 'util.js':
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'util', locale))

        if path == 'app.js':
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'app', locale))

        if path.endswith('.css'):
            s = path.split('.')
            if len(s) != 2:
                raise gws.NotFoundError(f'invalid css request: {p.path=}')
            content = gws.base.client.bundles.css(self.root, 'app', s[0])
            if not content:
                raise gws.NotFoundError(f'invalid css request: {p.path=}')
            return gws.ContentResponse(mime=gws.lib.mime.CSS, content=content)

        raise gws.NotFoundError(f'invalid system asset: {p.path=}')

    def _serve_path(self, req: gws.WebRequester, p: AssetRequest):
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

        if project_assets:
            real_path = gws.lib.osx.abs_web_path(req_path, project_assets.dir)
        if not real_path and site_assets:
            real_path = gws.lib.osx.abs_web_path(req_path, site_assets.dir)
        if not real_path:
            raise gws.NotFoundError(f'no real path for {req_path=}')

        tpl = self.root.app.templateMgr.template_from_path(real_path)
        if tpl:
            locale = gws.lib.intl.locale(p.localeUid, project.localeUids if project else self.root.app.localeUids)
            return self._serve_template(req, tpl, project, locale)

        mime = gws.lib.mime.for_path(real_path)

        if not _valid_mime_type(mime, project_assets, site_assets):
            # NB: pretend the file doesn't exist
            raise gws.NotFoundError(f'invalid mime path={real_path!r} mime={mime!r}')

        gws.log.debug(f'serving {real_path!r} for {req_path!r}')
        return gws.ContentResponse(contentPath=real_path, mime=mime)

    def _serve_template(self, req: gws.WebRequester, tpl: gws.Template, project: Optional[gws.Project], locale: gws.Locale):
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

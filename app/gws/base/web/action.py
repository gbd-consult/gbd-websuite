"""Serve dynamic assets."""

import os
import re

import gws
import gws.base.action
import gws.base.client.bundles
import gws.base.template
import gws.base.web.error
import gws.lib.mime
import gws.types as t


class AssetRequest(gws.Request):
    path: str


class AssetResponse(gws.Request):
    content: str
    mime: str


@gws.ext.object.action('web')
class Object(gws.base.action.Object):
    """Web action"""

    @gws.ext.command.api('webAsset')
    def api_asset(self, req: gws.IWebRequester, p: AssetRequest) -> AssetResponse:
        """Return an asset under the given path and project"""
        r = _serve_path(self.root, req, p)
        return AssetResponse(content=r.content, mime=r.mime)

    @gws.ext.command.get('webAsset')
    def http_asset(self, req: gws.IWebRequester, p: AssetRequest) -> gws.ContentResponse:
        return _serve_path(self.root, req, p)

    @gws.ext.command.get('webDownload')
    def download(self, req: gws.IWebRequester, p) -> gws.ContentResponse:
        return _serve_path(self.root, req, p, as_attachment=True)

    @gws.ext.command.get('webSystemAsset')
    def sys_asset(self, req: gws.IWebRequester, p: AssetRequest) -> gws.ContentResponse:
        locale_uid = p.localeUid or self.root.app.localeUids[0]

        # eg. '8.0.0.light.css, 8.0.0.vendor.js etc

        if p.path.endswith('.vendor.js'):
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'vendor', locale_uid))

        if p.path.endswith('.util.js'):
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'util', locale_uid))

        if p.path.endswith('.app.js'):
            return gws.ContentResponse(
                mime=gws.lib.mime.JS,
                content=gws.base.client.bundles.javascript(self.root, 'app', locale_uid))

        if p.path.endswith('.css'):
            theme = p.path.split('.')[-2]
            return gws.ContentResponse(
                mime=gws.lib.mime.CSS,
                content=gws.base.client.bundles.css(self.root, 'app', theme))


def _serve_path(root: gws.IRoot, req: gws.IWebRequester, p: AssetRequest, as_attachment=False):
    spath = str(p.get('path') or '')
    if not spath:
        raise gws.base.web.error.NotFound()

    site_assets = req.site.assetsRoot

    project = None
    project_assets = None

    project_uid = p.get('projectUid')
    if project_uid:
        project = req.require_project(project_uid)
        project_assets = project.assetsRoot

    rpath = None

    if project_assets:
        rpath = _abs_path(spath, project_assets.dir)
    if not rpath and site_assets:
        rpath = _abs_path(spath, site_assets.dir)
    if not rpath:
        raise gws.base.web.error.NotFound()

    locale_uid = p.localeUid
    if project and locale_uid not in project.localeUids:
        locale_uid = project.localeUids[0]

    tpl = gws.base.template.from_path(root, rpath)

    if tpl:
        # give the template an empty response to manipulate (e.g. add 'location')
        res = gws.ContentResponse()
        args = {
            'project': project,
            'projects': _projects_for_user(root, req.user),
            'request': req,
            'user': req.user,
            'params': p,
            'response': res,
            'localeUid': locale_uid,
        }

        render_res = tpl.render(gws.TemplateRenderInput(args=args))

        if gws.is_empty(res):
            res = render_res

        return res

    mime = gws.lib.mime.for_path(rpath)

    if not _valid_mime_type(mime, project_assets, site_assets):
        gws.log.error(f'invalid mime path={rpath!r} mime={mime!r}')
        # NB: pretend the file doesn't exist
        raise gws.base.web.error.NotFound()

    gws.log.debug(f'serving {rpath!r} for {spath!r}')

    return gws.ContentResponse(mime=mime, path=rpath, asAttachment=as_attachment)


def _projects_for_user(root, user):
    ps = [
        p
        for p in root.app.projects.values()
        if user.can_use(p)
    ]
    return sorted(ps, key=lambda p: p.title)


_dir_re = r'^[A-Za-z0-9_]+$'
_fil_re = r'^[A-Za-z0-9_]+(\.[a-z0-9]+)*$'


def _abs_path(path, basedir):
    gws.log.debug(f'trying {path!r} in {basedir!r}')

    dirs = []
    for s in path.split('/'):
        s = s.strip()
        if s:
            dirs.append(s)

    fname = dirs.pop()

    if not all(re.match(_dir_re, p) for p in dirs):
        gws.log.error(f'invalid dirname in path={path!r}')
        return

    if not re.match(_fil_re, fname):
        gws.log.error(f'invalid filename in path={path!r}')
        return

    p = basedir + '/' + '/'.join(dirs) + '/' + fname

    if not os.path.isfile(p):
        gws.log.error(f'not a file path={path!r}')
        return

    return p


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


def _valid_mime_type(mt, project_assets: t.Optional[gws.WebDocumentRoot], site_assets: t.Optional[gws.WebDocumentRoot]):
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

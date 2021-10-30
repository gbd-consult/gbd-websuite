"""Serve dynamic assets."""

import os
import re

import gws
import gws.base.api
import gws.base.client.bundles
import gws.base.template
import gws.base.web.error
import gws.lib.mime
import gws.types as t


class GetAssetParams(gws.Params):
    path: str


class GetAssetResponse(gws.Params):
    content: str
    mime: str


class SysAssetParams(gws.Params):
    path: str
    version: str = ''


@gws.ext.Object('action.web')
class Object(gws.base.api.action.Object):
    """Web action"""

    @gws.ext.command('api.web.asset')
    def api_asset(self, req: gws.IWebRequest, p: GetAssetParams) -> GetAssetResponse:
        """Return an asset under the given path and project"""
        r = _serve_path(self.root, req, p)
        return GetAssetResponse(content=r.content, mime=r.mime)

    @gws.ext.command('get.web.asset')
    def http_asset(self, req: gws.IWebRequest, p: GetAssetParams) -> gws.ContentResponse:
        return _serve_path(self.root, req, p)

    @gws.ext.command('get.web.download')
    def download(self, req: gws.IWebRequest, p) -> gws.ContentResponse:
        return _serve_path(self.root, req, p, True)

    @gws.ext.command('get.web.sysAsset')
    def sys_asset(self, req: gws.IWebRequest, p: SysAssetParams) -> gws.ContentResponse:
        locale_uid = p.localeUid or self.root.application.locale_uids[0]

        # eg. '8.0.0.light.css, 8.0.0.vendor.js etc

        if p.path.endswith('.vendor.js'):
            return gws.ContentResponse(
                content=gws.base.client.bundles.javascript(self.root, 'vendor', locale_uid),
                mime=gws.lib.mime.JS)

        if p.path.endswith('.util.js'):
            return gws.ContentResponse(
                content=gws.base.client.bundles.javascript(self.root, 'util', locale_uid),
                mime=gws.lib.mime.JS)

        if p.path.endswith('.app.js'):
            return gws.ContentResponse(
                content=gws.base.client.bundles.javascript(self.root, 'app', locale_uid),
                mime=gws.lib.mime.JS)

        if p.path.endswith('.css'):
            theme = p.path.split('.')[-2]
            return gws.ContentResponse(
                content=gws.base.client.bundles.css(self.root, 'app', theme),
                mime=gws.lib.mime.CSS)


def _serve_path(root: gws.IRoot, req: gws.IWebRequest, p: GetAssetParams, as_attachment=False):
    spath = str(p.get('path') or '')
    if not spath:
        raise gws.base.web.error.NotFound()

    site_assets = req.site.assets_root

    project = None
    project_assets = None

    project_uid = p.get('projectUid')
    if project_uid:
        project = req.require_project(project_uid)
        project_assets = project.assets_root

    rpath = None

    if project_assets:
        rpath = _abs_path(spath, project_assets.dir)
    if not rpath and site_assets:
        rpath = _abs_path(spath, site_assets.dir)
    if not rpath:
        raise gws.base.web.error.NotFound()

    locale_uid = p.localeUid
    if project and locale_uid not in project.locale_uids:
        locale_uid = project.locale_uids[0]

    tpl = gws.base.template.create_from_path(root, rpath)

    if tpl:
        # give the template an empty response to manipulate (e.g. add 'location')
        res = gws.ContentResponse()
        context = {
            'project': project,
            'projects': _projects_for_user(root, req.user),
            'request': req,
            'user': req.user,
            'params': p,
            'response': res,
            'localeUid': locale_uid,
        }

        out = tpl.render(context, gws.TemplateRenderArgs())

        if gws.is_empty(res):
            res.mime = out.mime
            res.content = out.content

        return res

    mime = gws.lib.mime.for_path(rpath)

    if not _valid_mime_type(mime, project_assets, site_assets):
        gws.log.error(f'invalid mime path={rpath!r} mime={mime!r}')
        # NB: pretend the file doesn't exist
        raise gws.base.web.error.NotFound()

    gws.log.debug(f'serving {rpath!r} for {spath!r}')

    return gws.ContentResponse(mime=mime, path=rpath, as_attachment=as_attachment)


def _projects_for_user(root, user):
    ps = [
        p
        for p in root.find_all('gws.base.project')
        if user.can_use(p)
    ]
    return sorted(ps, key=lambda p: p.title)


_dir_re = r'^[A-Za-z0-9_]+$'
_file_re = r'^[A-Za-z0-9_]+(\.[a-z0-9]+)*$'


def _abs_path(path, basedir):
    gws.log.debug(f'trying {path!r} in {basedir!r}')

    parts = []
    for s in path.split('/'):
        s = s.strip()
        if s:
            parts.append(s)

    if not all(re.match(_dir_re, p) for p in parts[:-1]):
        gws.log.error(f'invalid dirname in path={path!r}')
    if not re.match(_file_re, parts[-1]):
        gws.log.error(f'invalid filename in path={path!r}')

    p = basedir + '/' + '/'.join(parts)

    if not os.path.isfile(p):
        gws.log.error(f'not a file path={path!r}')
        return None

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


def _valid_mime_type(mt, project_assets: t.Optional[gws.DocumentRoot], site_assets: t.Optional[gws.DocumentRoot]):
    if project_assets and project_assets.allow_mime:
        return mt in project_assets.allow_mime
    if site_assets and site_assets.allow_mime:
        return mt in site_assets.allow_mime
    if mt not in _DEFAULT_ALLOWED_MIME_TYPES:
        return False
    if project_assets and project_assets.deny_mime:
        return mt not in project_assets.deny_mime
    if site_assets and site_assets.deny_mime:
        return mt not in site_assets.deny_mime
    return True

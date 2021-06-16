"""Serve dynamic assets."""

import os

import gws
import gws.base.action
import gws.base.template
import gws.config
import gws.server
import gws.lib.job
import gws.lib.mime
import gws.lib.os2
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Asset generation action"""
    pass


class GetPathParams(t.Params):
    path: str


class GetResultParams(t.Params):
    jobUid: str


class Object(gws.base.action.Object):

    def api_get(self, req: t.IRequest, p: GetPathParams) -> t.HttpResponse:
        """Return an asset under the given path and project"""
        return self._serve_path(req, p)

    def http_get_path(self, req: t.IRequest, p: GetPathParams) -> t.HttpResponse:
        return self._serve_path(req, p)

    def http_get_download(self, req: t.IRequest, p) -> t.HttpResponse:
        return self._serve_path(req, p, True)

    def _serve_path(self, req: t.IRequest, p: GetPathParams, as_attachment=False):
        spath = str(p.get('path') or '')
        if not spath:
            raise gws.web.error.NotFound()

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
            raise gws.web.error.NotFound()

        tpl = gws.base.template.from_path(self.root, rpath)

        if tpl:
            locale_uid = p.localeUid
            if project and locale_uid not in project.locale_uids:
                locale_uid = project.locale_uids[0]

            # give the template an empty response to manipulate (e.g. add 'location')
            r = t.HttpResponse()
            context = {
                'project': project,
                'projects': _projects_for_user(req.user),
                'request': req,
                'user': req.user,
                'params': p,
                'response': r,
                'localeUid': locale_uid,
            }

            out = tpl.render(context)

            if gws.is_empty(r):
                r.mime = out.mime
                r.content = out.content

            return r

        mime = gws.lib.mime.for_path(rpath)

        if not _valid_mime_type(mime, project_assets, site_assets):
            gws.log.error(f'invalid mime path={rpath!r} mime={mime!r}')
            # NB: pretend the file doesn't exist
            raise gws.web.error.NotFound()

        gws.log.debug(f'serving {rpath!r} for {spath!r}')

        attachment_name = None

        if as_attachment:
            p = gws.lib.os2.parse_path(spath)
            attachment_name = p['name'] + '.' + gws.lib.mime.extension(mime)

        return t.FileResponse(mime=mime, path=rpath, attachment_name=attachment_name)


def _projects_for_user(user):
    ps = [
        p
        for p in gws.config.root().find_all('gws.base.project')
        if user.can_use(p)
    ]
    return sorted(ps, key=lambda p: p.title)


def _abs_path(path, basedir):
    gws.log.debug(f'trying {path!r} in {basedir!r}')
    p = path.strip('/')

    if p.startswith('.') or '/.' in p:
        gws.log.error(f'dotted path={path!r}')
        return None

    p = os.path.abspath(os.path.join(basedir, p))
    if not os.path.isfile(p):
        gws.log.error(f'not a file path={path!r}')
        return None

    if not p.startswith(basedir):
        gws.log.error(f'invalid path={path!r}')
        return None

    return p


def _valid_mime_type(mt, project_assets: t.DocumentRoot, site_assets: t.DocumentRoot):
    if project_assets and project_assets.allow_mime:
        return mt in project_assets.allow_mime
    if site_assets and site_assets.allow_mime:
        return mt in site_assets.allow_mime
    if mt not in gws.lib.mime.DEFAULT_ALLOWED:
        return False
    if project_assets and project_assets.deny_mime:
        return mt not in project_assets.deny_mime
    if site_assets and site_assets.deny_mime:
        return mt not in site_assets.deny_mime
    return True

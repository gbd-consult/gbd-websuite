"""Serve dynamic assets."""

import os

import gws
import gws.common.action
import gws.common.template
import gws.config
import gws.server
import gws.tools.job
import gws.tools.mime
import gws.tools.os2
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Asset generation action"""
    pass


class GetPathParams(t.Params):
    path: str


class GetResultParams(t.Params):
    jobUid: str


class Object(gws.common.action.Object):

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

        tpl = gws.common.template.from_path(self.root, rpath)

        if tpl:
            # give the template an empty response to manipulate (e.g. add 'location')
            r = t.HttpResponse()
            context = {
                'project': project,
                'projects': _projects_for_user(req.user),
                'request': req,
                'user': req.user,
                'params': p,
                'response': r,
            }

            out = tpl.render(context)

            if gws.is_empty(r):
                r.mime = out.mime
                r.content = out.content

            return r

        mime = gws.tools.mime.for_path(rpath)

        if not _valid_mime_type(mime, project_assets, site_assets):
            gws.log.error(f'invalid mime path={rpath!r} mime={mime!r}')
            # NB: pretend the file doesn't exist
            raise gws.web.error.NotFound()

        gws.log.info(f'serving {rpath!r} for {spath!r}')

        attachment_name = None

        if as_attachment:
            p = gws.tools.os2.parse_path(spath)
            attachment_name = p['name'] + '.' + gws.tools.mime.extension(mime)

        return t.FileResponse(mime=mime, path=rpath, attachment_name=attachment_name)


def _projects_for_user(user):
    ps = [
        p
        for p in gws.config.root().find_all('gws.common.project')
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
    if mt not in gws.tools.mime.default_allowed:
        return False
    if project_assets and project_assets.deny_mime:
        return mt not in project_assets.deny_mime
    if site_assets and site_assets.deny_mime:
        return mt not in site_assets.deny_mime
    return True

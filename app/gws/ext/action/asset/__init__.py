import os

import gws
import gws.config
import gws.server
import gws.web.error
import gws.tools.mime
import gws.tools.job

import gws.types as t
import gws.common.template


class Config(t.WithTypeAndAccess):
    """Asset generation action"""
    pass


class GetPathParams(t.Params):
    path: str

class GetResultParams(t.Params):
    jobUid: str


class Object(gws.ActionObject):

    def api_get(self, req, p: GetPathParams) -> t.HttpResponse:
        """Return an asset under the given path and project"""
        return self._serve_path(req, p)

    def http_get_path(self, req, p: GetPathParams) -> t.HttpResponse:
        return self._serve_path(req, p)

    def http_get_download(self, req, p) -> t.HttpResponse:
        # @TODO
        pass

    def http_get_result(self, req, p: GetResultParams) -> t.HttpResponse:
        job = gws.tools.job.get_for(req.user, p.jobUid)
        if not job or job.state != gws.tools.job.State.complete:
            raise gws.web.error.NotFound()
        with open(job.result, 'rb') as fp:
            content = fp.read()
        return t.HttpResponse({
            'mimeType': gws.tools.mime.for_path(job.result),
            'content': content
        })

    def _serve_path(self, req, p):
        site_assets = req.site.assets_root

        project = None
        project_assets = None

        project_uid = p.get('projectUid')
        if project_uid:
            project = req.require_project(project_uid)
            project_assets = project.assets_root

        spath = str(p.get('path') or '')
        rpath = None

        if project_assets:
            rpath = _abs_path(spath, project_assets.dir)
        if not rpath and site_assets:
            rpath = _abs_path(spath, site_assets.dir)
        if not rpath:
            raise gws.web.error.NotFound()

        gws.log.info(f'serving {rpath!r} for {spath!r}')

        template_type = gws.common.template.type_from_path(rpath)
        if template_type:
            tpl = self.create_shared_object('gws.ext.template', rpath, t.Config({
                'type': template_type,
                'path': rpath
            }))
            context = _default_template_context(req, project)
            context['params'] = p
            tr = tpl.render(context)
            # @TODO handle path
            return t.HttpResponse({
                'mimeType': tr.mimeType,
                'content': tr.content
            })

        mt = gws.tools.mime.for_path(rpath)

        # @TODO check project mime types config

        if mt not in gws.tools.mime.default_allowed:
            gws.log.error(f'invalid mime path={rpath!r} mt={mt!r}')
            raise gws.web.error.NotFound()

            # @TODO optimize streaming

        with open(rpath, 'rb') as fp:
            s = fp.read()
        return t.HttpResponse({
            'mimeType': mt,
            'content': s
        })


def _projects_for_user(user):
    ps = [
        p
        for p in gws.config.find_all('gws.common.project')
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


def _default_template_context(req, project):
    return {
        'project': project,
        'projects': _projects_for_user(req.user),
        'request': req,
        'user': req.user,
    }

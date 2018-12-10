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
    """asset (dynamic html) action"""
    pass


class AssetParams(t.Data):
    path: str
    projectUid: t.Optional[str]


class Object(gws.Object):
    def api_get(self, req, p: AssetParams) -> t.HttpResponse:
        """Return an asset under the given path and project"""
        return self._serve_path(req, p)

    def http_get_path(self, req, _) -> t.HttpResponse:
        return self._serve_path(req, req.params)

    def http_get_download(self, req, p) -> t.HttpResponse:
        # @TODO
        pass

    def http_get_result(self, req, _) -> t.HttpResponse:
        job = gws.tools.job.get_for(req.user, req.param('jobUid'))
        if not job or job.state != gws.tools.job.State.complete:
            raise gws.web.error.NotFound()
        with open(job.result, 'rb') as fp:
            content = fp.read()
        return t.HttpResponse({
            'mimeType': gws.tools.mime.for_path(job.result),
            'content': content
        })


    def _serve_path(self, req, p):
        site_assets_config = req.site.get('assets')

        project = None
        project_assets_config = None

        project_uid = p.get('projectUid')
        if project_uid:
            project = req.require_project(project_uid)
            project_assets_config = project.var('assets', parent=True)

        spath = str(p.get('path') or '')
        rpath = None

        if project_assets_config:
            rpath = _abs_path(spath, project_assets_config.dir)
        if not rpath and site_assets_config:
            rpath = _abs_path(spath, site_assets_config.dir)
        if not rpath:
            raise gws.web.error.NotFound()

        gws.log.info(f'serving "{rpath}" for "{spath}"')

        template_type = gws.common.template.type_from_path(rpath)
        if template_type:
            tpl = self.create_shared_object('gws.ext.template', rpath, t.Config({
                'type': template_type,
                'path': rpath
            }))
            tr = tpl.render(_default_template_context(req, project))
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
        if user.can_read(p)
    ]
    return sorted(ps, key=lambda p: p.title)


def _abs_path(path, basedir):
    gws.log.debug(f'trying {path!r} in {basedir!r}')
    p = path.lstrip('/')
    if p.endswith('/'):
        p = p[:-1]

    if p.startswith('.') or '/.' in p:
        gws.log.error(f'invalid path={path!r}')
        return None

    p = os.path.abspath(os.path.join(basedir, p))
    if os.path.isfile(p):
        gws.log.debug(f'{path} => {p}')
        return p

    gws.log.info(f'file not found path={path!r}')
    return None


def _default_template_context(req, project):
    return {
        'project': project,
        'projects': _projects_for_user(req.user),
        'request': req,
        'user': req.user,
        'gws': {
            'version': gws.VERSION,
        }
    }

from typing import Optional, cast

import re
import os
import hashlib

import gws
import gws.base.job
import gws.base.shape
import gws.base.action
import gws.lib.mime
import gws.lib.jsonx
import gws.lib.datetimex as dtx
import gws.lib.osx as osx

from . import core, packager, patcher, api, caps

gws.ext.new.action('qfieldcloud')


class Config(gws.ConfigWithAccess):
    projects: list[core.ProjectConfig]
    """QField Cloud projects."""


class Props(gws.base.action.Props):
    pass


class Request(gws.Data):
    req: gws.WebRequester
    """The original web request."""
    route: str
    """Request method and path."""
    parts: dict
    """Variables from the path."""
    qs: dict
    """Query string parameters."""
    post: dict
    """POST payload."""
    project: gws.Project
    """GWS Project context."""
    qfcProject: core.QfcProject
    """QField Cloud Project context."""
    user: gws.User
    """Authenticated user."""
    sess: gws.AuthSession
    """Authentication session."""
    token: str
    """Authentication token."""


class WorkerPayload(gws.Data):
    actionUid: str
    jobType: str
    qfcProjectUid: str
    projectUid: str


def route(pattern: str):
    def decorator(fn):
        fn._route_pattern = pattern
        return fn

    return decorator


class Object(gws.base.action.Object):
    qfcProjects: list[core.QfcProject]
    capsCache: dict[str, caps.Caps]

    def configure(self):
        self.qfcProjects = []
        for p in self.cfg('projects') or []:
            qp = self.create_child(core.QfcProject, p)
            if qp:
                self.qfcProjects.append(cast(core.QfcProject, qp))

    def __getstate__(self):
        return gws.u.omit(vars(self), 'capsCache')

    @gws.ext.command.raw('qfieldcloudApi')
    def raw_request(self, req: gws.WebRequester, p: gws.Request) -> gws.ContentResponse:
        path = req.path().strip('/')
        if not path:
            raise gws.NotFoundError('API path not specified')

        path_parts = path.split('/')
        project = cast(gws.Project, self.find_closest(gws.ext.object.project))

        if path_parts[0] == 'projectUid':
            try:
                path_parts.pop(0)
                uid = path_parts.pop(0)
                path = '/'.join(path_parts)
            except IndexError:
                raise gws.NotFoundError('gws project UID not specified')
            if not project:
                project = req.user.require_project(uid)
            elif uid != project.uid:
                raise gws.NotFoundError(f'gws project UID mismatch: {uid=} != {project.uid=}')

        if not project:
            raise gws.NotFoundError('gws project not found')

        path = path.strip('/')
        route = f'{req.method} {path}'

        for name in dir(self):
            fn = getattr(self, name)
            if callable(fn) and hasattr(fn, '_route_pattern'):
                m = re.match(f'^{fn._route_pattern}$', route)
                if m:
                    rx = Request(
                        req=req,
                        project=project,
                        route=route,
                        parts=m.groupdict(),
                        post={},
                        qs=req.query_params(),
                    )
                    return self._handle_route(fn, rx)

        raise gws.NotFoundError(f'API {route=} not found')

    _public_routes = [
        'GET api/v1/auth/providers',
        'POST api/v1/auth/token',
    ]

    def _handle_route(self, fn, rx: Request) -> gws.ContentResponse:
        if rx.req.isApi:
            rx.post = rx.req.struct()
        elif rx.req.isForm:
            rx.post = dict(rx.req.form())
        elif rx.req.isPost:
            rx.post = dict(raw=rx.req.data())

        gws.log.debug(f'API_REQUEST {rx.route=} -> {fn.__name__} {rx=}')

        if rx.route not in self._public_routes:
            self.authorize_from_token(rx)

        res = fn(rx)

        if not res:
            return gws.ContentResponse(content='')

        if isinstance(res, gws.ContentResponse):
            return res

        if isinstance(res, (list, dict, gws.Data)):
            return gws.ContentResponse(
                content=gws.lib.jsonx.to_string(res),
                mime=gws.lib.mime.JSON,
            )

        raise gws.Error(f'API {rx.route=} invalid response type: {type(res)}')

    ##

    @route('POST api/v1/auth/logout')
    def on_post_auth_logout(self, rx: Request):
        am = self.root.app.authMgr
        am.sessionMgr.delete(rx.sess)
        gws.log.debug(f'{self=} {rx=}')

    @route('GET api/v1/auth/providers')
    def on_get_auth_providers(self, rx: Request) -> list[api.AuthProvider]:
        return [
            api.AuthProvider(type='credentials', id='credentials', name='Username / Password'),
        ]

    @route('POST api/v1/auth/token')
    def on_post_auth_token(self, rx: Request) -> api.AuthToken:
        self.authorize_from_credentials(
            gws.Data(
                username=rx.post.get('username', ''),
                password=rx.post.get('password', ''),
            ),
            rx,
        )
        am = self.root.app.authMgr
        return api.AuthToken(
            token=rx.token,
            expires_at=dtx.to_iso_string(dtx.add(seconds=am.sessionMgr.lifeTime)),
            username=rx.user.loginName,
            type=api.UserType.person,
            full_name=rx.user.displayName,
            avatar_url='',
            email='',
            first_name='',
            last_name='',
        )

    @route('GET api/v1/auth/user')
    def on_get_auth_user(self, rx: Request) -> api.CompleteUser:
        return api.CompleteUser(
            username=rx.user.loginName,
            type=api.UserType.person,
            full_name=rx.user.displayName,
            avatar_url='',
            email='',
            first_name='',
            last_name='',
        )

    @route('GET api/v1/projects')
    def on_get_projects(self, rx: Request) -> list[api.Project]:
        limit = int(rx.qs.get('limit', 100))
        offset = int(rx.qs.get('offset', 0))
        qps = self.get_qfc_projects(rx.project, rx.user)
        return [_format_project(qp, rx) for qp in qps[offset : offset + limit]]

    @route('GET api/v1/projects/(?P<project_id>[^/]+)')
    def on_get_projects_id(self, rx: Request) -> api.Project:
        self.set_qfc_project_from_parts(rx)
        return _format_project(rx.qfcProject, rx)

    @route('POST api/v1/jobs')
    def on_post_jobs(self, rx: Request) -> api.Job:
        project_id = rx.post.get('project_id', '')
        type = rx.post.get('type', '')
        self.set_qfc_project(project_id, rx)
        if type != api.TypeEnum.package:
            raise gws.Error(f'Unsupported job type: {type!r}')

        job = self.create_package_job(rx)
        return _format_job(job, rx)

    @route('GET api/v1/jobs/(?P<job_id>[^/]+)')
    def on_get_jobs_id(self, rx: Request) -> api.Job:
        job_id = rx.parts.get('job_id', '')
        job = self.get_job(job_id, rx.project, rx.user)
        if not job:
            raise gws.NotFoundError(f'Job {job_id!r} not found')
        return _format_job(job, rx)

    @route('GET api/v1/packages/(?P<project_id>[^/]+)/(?P<package_version>[^/]+)')
    def on_get_package(self, rx: Request) -> api.Package:
        self.set_qfc_project_from_parts(rx)

        # @TODO do we need versions?
        # @TODO do we need layers?
        # package_version = rx.parts.get('package_version', '')

        path_map = self.get_latest_package_path_map(rx)
        return api.Package(
            files=_format_files(path_map),
            layers=[],
            status=api.JobStatusEnum.finished,
            package_id=rx.qfcProject.uid,
            packaged_at=dtx.to_iso_string(),
            data_last_updated_at=dtx.to_iso_string(),
        )

    @route('GET api/v1/packages/(?P<project_id>[^/]+)/(?P<package_version>[^/]+)/files/(?P<file_name>.+)')
    def on_get_package_file(self, rx: Request) -> gws.ContentResponse:
        self.set_qfc_project_from_parts(rx)

        file_name = rx.parts.get('file_name', '')
        path_map = self.get_latest_package_path_map(rx)
        for fname, p in path_map.items():
            if file_name == fname:
                return gws.ContentResponse(contentPath=p)
        raise gws.NotFoundError(f'file {file_name!r} not found')

    @route('GET api/v1/files/(?P<project_id>[^/]+)')
    def on_get_files(self, rx: Request) -> list[api.PackageFile]:
        self.set_qfc_project_from_parts(rx)
        path_map = self.get_latest_package_path_map(rx)
        return _format_files(path_map)

    @route('POST api/v1/deltas/(?P<project_id>[^/]+)')
    def on_post_deltas(self, rx: Request):
        self.set_qfc_project_from_parts(rx)

        # deltas come as a multipart file upload
        try:
            js = gws.lib.jsonx.from_string(rx.post['file'].stream.read().decode('utf-8'))
            payload = api.DeltasPayload(
                deltas=js['deltas'],
                files=js.get('files', []),
                id=js['id'],
                project=js['project'],
                version=js['version'],
            )
        except Exception as exc:
            raise gws.BadRequestError(f'invalid delta file content: {exc}')

        self.store_delta_payload(payload, rx)

        changes = []
        for d in payload.deltas:
            new = d.get('new', {})
            old = d.get('old', {})
            chg = patcher.Change(
                uid=d['uuid'],
                type=d['method'],
                layerUid=d['localLayerId'],
                newAtts=new['attributes'] if new else {},
                oldAtts=old['attributes'] if old else {},
                wkt=new.get('geometry', ''),
            )
            changes.append(chg)

        args = patcher.Args(
            qfcProject=rx.qfcProject,
            caps=self.get_caps(rx.qfcProject),
            project=rx.project,
            user=rx.user,
            baseDir='',
            changes=changes,
        )
        self.get_patcher().apply_changes(self.root, args)

        self.set_delta_payload_applied(payload.id, rx)

    @route('GET api/v1/deltas/(?P<project_id>[^/]+)/(?P<payload_id>.+)')
    def on_get_deltas(self, rx: Request) -> list[api.StoredDelta]:
        self.set_qfc_project_from_parts(rx)

        # the content of the delta does not seem to matter much, only the ID and  status=applied
        # see QField/src/core/qfieldcloud/qfieldcloudproject.cpp : getDeltaStatus()

        payload_id = rx.parts.get('payload_id', '')
        sds = self.get_delta_payload(payload_id, rx)
        if not sds:
            raise gws.NotFoundError(f'delta {payload_id=} not found')
        return sds

    @route('POST api/v1/files/(?P<project_id>[^/]+)/(?P<path>.+)')
    def on_post_file(self, rx: Request):
        self.set_qfc_project_from_parts(rx)

        path = rx.parts.get('path', '')
        try:
            fc = rx.post['file'].stream.read()
        except Exception as exc:
            raise gws.BadRequestError(f'invalid file upload: {exc}')

        args = patcher.Args(
            qfcProject=rx.qfcProject,
            caps=self.get_caps(rx.qfcProject),
            project=rx.project,
            user=rx.user,
            baseDir='',
            filePath=path,
            fileContent=fc,
        )
        self.get_patcher().apply_upload(self.root, args)

    ##

    def get_packager(self) -> packager.Object:
        return packager.Object()

    def get_patcher(self) -> patcher.Object:
        return patcher.Object()

    ##

    def get_caps(self, qfc_project: core.QfcProject) -> caps.Caps:
        if not hasattr(self, 'capsCache'):
            self.capsCache = {}
        cs = self.get_cached_caps(qfc_project)
        if cs:
            return cs

        pa = caps.Parser(qfc_project)
        pa.parse()
        gws.u.serialize_to_path(pa.caps, f'{self.fs_project_cache_dir(qfc_project)}/caps.pickle')
        pa.create_models()
        pa.assign_path_props()

        self.capsCache[qfc_project.uid] = pa.caps
        gws.log.debug(f'get_caps: {qfc_project.uid=}: created')
        return pa.caps

    def get_cached_caps(self, qfc_project: core.QfcProject) -> Optional[caps.Caps]:
        cs = self.capsCache.get(qfc_project.uid)
        if not cs:
            gws.log.debug(f'get_caps: {qfc_project.uid=}: not found')
            return

        qp = qfc_project.qgisProvider.qgis_project()
        if qp.sourceHash != cs.sourceHash:
            gws.log.debug(f'get_caps: {qfc_project.uid=}: hash changed: {cs.sourceHash=} != {qp.sourceHash=}')
            self.capsCache.pop(qfc_project.uid, None)
            return

        gws.log.debug(f'get_caps: {qfc_project.uid=}: CACHED!')
        return cs

    def authorize_from_credentials(self, credentials: gws.Data, rx: Request):
        am = self.root.app.authMgr
        user = am.authenticate(cast(gws.AuthMethod, self), credentials)
        if not user:
            raise gws.ForbiddenError('invalid username or password')
        rx.sess = am.sessionMgr.create(cast(gws.AuthMethod, self), user)
        rx.user = user
        rx.token = rx.sess.uid

    def authorize_from_token(self, rx: Request):
        h = rx.req.header('Authorization', '')
        m = re.match(r'^Token (.+)$', h)
        if not m:
            raise gws.ForbiddenError('token_auth: missing or invalid Authorization header')
        token = m.group(1)
        am = self.root.app.authMgr
        sess = am.sessionMgr.get_valid(token)
        if not sess:
            raise gws.ForbiddenError(f'token_auth: invalid or expired {token=}')
        rx.sess = sess
        rx.user = sess.user
        rx.token = sess.uid
        gws.log.debug(f'token_auth: ok: {rx.token=} {rx.user.uid=} {rx.user.loginName=}')

    ##

    def set_qfc_project(self, uid: str, rx: Request):
        qp = self.get_qfc_project(uid, rx.project, rx.user)
        if not qp:
            raise gws.NotFoundError(f'project {uid!r} not found')
        rx.qfcProject = qp

    def set_qfc_project_from_parts(self, rx: Request):
        uid = rx.parts.get('project_id', '')
        self.set_qfc_project(uid, rx)

    ##

    def get_qfc_projects(self, project: gws.Project, user: gws.User) -> list[core.QfcProject]:
        return [p for p in self.qfcProjects if user.can_use(p)]

    def get_qfc_project(self, qfc_project_uid: str, project: gws.Project, user: gws.User) -> Optional[core.QfcProject]:
        for qp in self.get_qfc_projects(project, user):
            if qp.uid == qfc_project_uid:
                return qp

    ##

    def create_package_job(self, rx: Request) -> gws.Job:
        mgr = self.root.app.jobMgr
        p = WorkerPayload(
            actionUid=self.uid,
            jobType='package',
            qfcProjectUid=rx.qfcProject.uid,
            projectUid=rx.project.uid,
        )
        job = mgr.create_job(
            PackageWorker,
            rx.user,
            payload=gws.u.to_dict(p),
        )
        return mgr.schedule_job(job)

    def create_package_from_worker(self, worker: 'PackageWorker', pa: WorkerPayload):
        project = worker.user.require_project(pa.projectUid)
        qfc_project = gws.u.require(self.get_qfc_project(pa.qfcProjectUid, project, worker.user))

        self.fs_cleanup_old_packages(qfc_project)
        
        uid = dtx.to_basic_string(with_ms=True)
        pkg_dir = self.fs_new_package_dir(qfc_project, uid)
        args = packager.Args(
            uid=uid,
            qfcProject=qfc_project,
            caps=self.get_caps(qfc_project),
            project=project,
            user=worker.user,
            packageDir=pkg_dir,
            mapCacheDir=self.fs_project_cache_dir(qfc_project),
            withBaseMap=True,
            withData=True,
            withMedia=True,
            withQgis=True,
        )
        self.get_packager().create_package(self.root, args)

    def create_package_from_cli(self, qfc_project_uid: str, target_dir: str, project: gws.Project, user: gws.User):
        qfc_project = self.get_qfc_project(qfc_project_uid, project, user)
        if not qfc_project:
            raise gws.NotFoundError(f'project {qfc_project_uid!r} not found')

        args = packager.Args(
            uid='cli',
            qfcProject=qfc_project,
            caps=self.get_caps(qfc_project),
            project=project,
            user=user,
            packageDir=target_dir,
            mapCacheDir=self.fs_project_cache_dir(qfc_project),
            withBaseMap=True,
            withData=True,
            withMedia=True,
            withQgis=True,
        )
        self.get_packager().create_package(self.root, args)

    def get_job(self, job_id: str, project: gws.Project, user: gws.User) -> Optional[gws.Job]:
        return self.root.app.jobMgr.get_job(job_id, user=user)

    ##

    def store_delta_payload(self, payload: api.DeltasPayload, rx: Request):
        self.fs_cleanup_old_deltas(rx.qfcProject)
        sds = [
            api.StoredDelta(
                id=delta['uuid'],
                deltafile_id=payload.id,
                created_by=rx.user.loginName,
                created_at=dtx.to_iso_string(),
                updated_at=dtx.to_iso_string(),
                status='STATUS_PENDING',
                client_id=delta['clientId'],
                output=None,
                last_status='pending',
                last_feedback=None,
                content=delta,
            )
            for delta in payload.deltas
        ]
        gws.lib.jsonx.to_path(
            self.fs_delta_payload_path(rx.qfcProject, payload.id),
            sds,
        )

    def set_delta_payload_applied(self, payload_id: str, rx: Request):
        sds = self.get_delta_payload(payload_id, rx)
        if not sds:
            return
        for sd in sds:
            sd.status = 'STATUS_APPLIED'
            sd.last_status = 'applied'
            sd.updated_at = dtx.to_iso_string()
        gws.lib.jsonx.to_path(
            self.fs_delta_payload_path(rx.qfcProject, payload_id),
            sds,
        )

    def get_delta_payload(self, payload_id: str, rx: Request) -> Optional[list[api.StoredDelta]]:
        path = self.fs_delta_payload_path(rx.qfcProject, payload_id)
        if not os.path.exists(path):
            gws.log.warning(f'stored delta {payload_id=}: not found: {path=}')
            return
        try:
            sds = [api.StoredDelta(d) for d in gws.lib.jsonx.from_path(path)]
        except Exception as exc:
            gws.log.warning(f'stored delta {payload_id=}: failed to load {path=}: {exc}')
            return
        for sd in sds:
            if sd.created_by != rx.user.loginName:
                gws.log.warning(f'stored delta {payload_id=}: user mismatch: {sd.created_by=} != {rx.user.loginName=}')
                return
        return sds

    ##

    def fs_project_base_dir(self, qfc_project: core.QfcProject) -> str:
        return gws.u.ensure_dir(f'{gws.c.VAR_DIR}/qfieldcloud/projects/{qfc_project.uid}')

    def fs_latest_package_dir(self, qfc_project: core.QfcProject) -> Optional[str]:
        base_dir = self.fs_project_base_dir(qfc_project)
        for pkg in sorted(osx.find_directories(base_dir, deep=False), reverse=True):
            m = re.search(r'package_(\d+)', pkg)
            if m:
                return pkg

    def fs_new_package_dir(self, qfc_project: core.QfcProject, uid: str) -> str:
        base_dir = self.fs_project_base_dir(qfc_project)
        pkg_dir = gws.u.ensure_dir(f'{base_dir}/package_{uid}')
        return pkg_dir

    def fs_cleanup_old_packages(self, qfc_project: core.QfcProject, keep_seconds: int = 3600):
        base_dir = self.fs_project_base_dir(qfc_project)
        now = dtx.now().timestamp()
        for pkg in osx.find_directories(base_dir, deep=False):
            m = re.search(r'package_(\d+)', pkg)
            if not m:
                continue
            t = osx.file_mtime(pkg)
            if now - t > keep_seconds:
                gws.log.info(f'fs_cleanup_old_packages: removing old package: {pkg=}')
                osx.rmdir(pkg)

    def fs_project_cache_dir(self, qfc_project: core.QfcProject) -> str:
        base_dir = self.fs_project_base_dir(qfc_project)
        return gws.u.ensure_dir(f'{base_dir}/cache')

    def fs_project_deltas_dir(self, qfc_project: core.QfcProject) -> str:
        base_dir = self.fs_project_base_dir(qfc_project)
        return gws.u.ensure_dir(f'{base_dir}/deltas')

    def fs_delta_payload_path(self, qfc_project: core.QfcProject, payload_id: str) -> str:
        d = self.fs_project_deltas_dir(qfc_project)
        u = gws.u.to_uid(payload_id)
        return f'{d}/{u}.json'

    def fs_cleanup_old_deltas(self, qfc_project: core.QfcProject, keep_seconds: int = 3600):
        d = self.fs_project_deltas_dir(qfc_project)
        now = dtx.now().timestamp()
        for f in osx.find_files(d, deep=False):
            t = osx.file_mtime(f)
            if now - t > keep_seconds:
                gws.log.info(f'fs_cleanup_old_deltas: removing old delta: {f=}')
                osx.unlink(f)

    def get_latest_package_path_map(self, rx: Request) -> dict[str, str]:
        d = self.fs_latest_package_dir(rx.qfcProject)
        try:
            return gws.lib.jsonx.from_path(f'{d}/path_map.json')
        except Exception:
            return {}


##


class PackageWorker(gws.base.job.worker.Object):
    @classmethod
    def run(cls, root: gws.Root, job: gws.Job):
        w = cls(root, job.user, job)
        w.work()

    def work(self):
        self.update_job(state=gws.JobState.running)
        pa = WorkerPayload(gws.u.require(self.get_job()).payload)
        action = cast(Object, self.root.get(pa.actionUid))
        action.create_package_from_worker(self, pa)
        self.update_job(state=gws.JobState.complete)


##


_DATE_CREATED = '2025-10-10T14:00:00'


def _format_project(qp: core.QfcProject, rx: Request) -> api.Project:
    return api.Project(
        id=qp.uid,
        name=qp.title,
        owner=rx.user.loginName,
        description='',
        private=True,
        is_public=False,
        created_at=_DATE_CREATED,
        updated_at=dtx.to_iso_string(),
        data_last_packaged_at=None,
        data_last_updated_at=dtx.to_iso_string(),
        can_repackage=True,
        needs_repackaging=True,
        status='ok',
        user_role='admin',
        user_role_origin='project_owner',
        shared_datasets_project_id=None,
        is_shared_datasets_project=False,
        is_featured=False,
        is_attachment_download_on_demand=False,
    )


def _format_files(path_map: dict[str, str]):
    return [
        api.PackageFile(
            name=fname,
            size=osx.file_size(p),
            uploaded_at=_get_time_iso(p),
            is_attachment=False,
            md5sum=_get_md5sum(p),
            last_modified=_get_time_iso(p),
            sha256=_get_sha256(p),
        )
        for fname, p in path_map.items()
    ]


def _format_job(job: gws.Job, rx: Request) -> api.Job:
    status_map = {
        gws.JobState.open: api.JobStatusEnum.pending,
        gws.JobState.running: api.JobStatusEnum.started,
        gws.JobState.complete: api.JobStatusEnum.finished,
        gws.JobState.error: api.JobStatusEnum.failed,
    }

    return api.Job(
        id=job.uid,
        type=job.payload.get('job_type', ''),
        created_at=dtx.to_iso_string(job.timeCreated),
        created_by=1,
        project_id=job.payload.get('project_uid', ''),
        status=status_map.get(job.state, api.JobStatusEnum.pending),
        updated_at=dtx.to_iso_string(job.timeUpdated),
        started_at=dtx.to_iso_string(job.timeUpdated) if job.state == gws.JobState.running else None,
        finished_at=dtx.to_iso_string(job.timeUpdated) if job.state == gws.JobState.complete else None,
    )


def _get_sha256(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _get_md5sum(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def _get_time_iso(path: str) -> str:
    t = osx.file_mtime(path)
    return dtx.to_iso_string(dtx.from_timestamp(t))

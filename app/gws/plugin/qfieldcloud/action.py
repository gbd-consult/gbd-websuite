"""QField Cloud API action."""

from typing import Optional, cast

import re
import os
import hashlib

import gws
import gws.base.auth
import gws.base.job
import gws.lib.shape
import gws.base.action
import gws.lib.mime
import gws.lib.jsonx
import gws.lib.datetimex as dtx
import gws.lib.osx as osx

from . import core, packager, patcher, api, caps


@gws.ext.config.action('qfieldcloud')
class Config(gws.ConfigWithAccess):
    """Endpoint for QField clients that emulates the QFieldCloud API."""

    projects: list[core.ProjectConfig]
    """Projects offered to QField clients."""
    auth: Optional[gws.base.auth.method.Config]
    """Token authentication method for QField clients."""


@gws.ext.props.action('qfieldcloud')
class Props(gws.base.action.Props):
    pass


class Request(gws.Data):
    """An API request, passed to the route handlers."""

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
    project: Optional[gws.Project]
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
    """Job payload for the package worker."""

    actionUid: str
    """Uid of the action."""
    jobType: str
    """Job type, ``package``."""
    qfcProjectUid: str
    """QField project uid."""
    projectUid: Optional[str]
    """GWS project uid."""


def route(pattern: str):
    """Decorator that marks a method as an API route handler.

    Args:
        pattern: Regular expression matched against ``"<METHOD> <path>"``. Named groups are passed to the handler in ``Request.parts``.

    Returns:
        The decorator.
    """
    def decorator(fn):
        fn._route_pattern = pattern
        return fn

    return decorator


@gws.ext.object.action('qfieldcloud')
class Object(gws.base.action.Object):
    """QField Cloud API action.

    Emulates the QFieldCloud API for the QField app: authenticates clients by
    token, lists the configured QField projects, creates packages in background
    jobs and applies deltas and file uploads from the devices.
    """

    qfcProjects: list[core.QfcProject]
    """Configured QField projects."""
    capsCache: dict[str, caps.Caps]
    """Capabilities by QField project uid. Not serialized."""
    method: gws.AuthMethod
    """Authorization method for QField clients."""

    def configure(self):
        self.method = cast(gws.AuthMethod, self.create_child(gws.ext.object.authMethod, self.cfg('auth'), type='qfieldcloud'))
        self.root.app.authMgr.add_method(self.method)
        self.qfcProjects = []
        for p in self.cfg('projects') or []:
            qp = self.create_child(core.QfcProject, p)
            if qp:
                self.qfcProjects.append(cast(core.QfcProject, qp))

    def __getstate__(self):
        """Return the state for pickling, without the caps cache."""
        return gws.u.omit(vars(self), 'capsCache')

    @gws.ext.command.raw('qfieldcloudApi')
    def raw_request(self, req: gws.WebRequester, p: gws.Request) -> gws.ContentResponse:
        """Handle a QFieldCloud API request."""
        try:
            return self.dispatch_request(req, p)
        except gws.NotFoundError as exc:
            return _error_response(404, 'object_not_found', exc)
        except gws.AuthenticationError as exc:
            return _error_response(401, 'authentication_failed', exc)
        except gws.ForbiddenError as exc:
            return _error_response(403, 'permission_denied', exc)
        except gws.BadRequestError as exc:
            return _error_response(400, 'validation_error', exc)

    def dispatch_request(self, req: gws.WebRequester, p: gws.Request) -> gws.ContentResponse:
        """Find the route handler for a request and call it.

        The path can start with ``projectUid/<uid>`` to set the GWS project.
        If the action belongs to a project, that project is used instead.

        Args:
            req: Web requester.
            p: Command request.

        Returns:
            The response.

        Raises:
            ``gws.NotFoundError``: If the path is empty, the GWS project is not found or no route matches.
            ``gws.ForbiddenError``: If the user cannot use the GWS project.
        """
        path = req.path().strip('/')
        if not path:
            raise gws.NotFoundError('API path not specified')

        path_parts = path.split('/')
        uid = p.get('projectUid')

        if path_parts[0] == 'projectUid':
            if len(path_parts) < 2:
                raise gws.NotFoundError('gws project UID not specified')
            uid = path_parts[1]
            path = '/'.join(path_parts[2:])

        project = cast(Optional[gws.Project], self.find_closest(gws.ext.object.project))
        project_uid = project.uid if project else uid
        if project_uid:
            project = req.user.require_project(project_uid)

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

    _public_routes = (
        'GET api/v1/auth/providers',
        'POST api/v1/auth/token',
        'GET api/v1/server/info',
        'GET api/v1/status',
    )

    def _handle_route(self, fn, rx: Request) -> gws.ContentResponse:
        """Read the request payload, authorize non-public routes, call the handler and convert its result to a response."""
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

        if res is None:
            return gws.ContentResponse(content='')

        if isinstance(res, gws.ContentResponse):
            return res

        if isinstance(res, (list, dict, gws.Data)):
            return gws.ContentResponse(
                content=gws.lib.jsonx.to_string(res),
                mimeType=gws.lib.mime.JSON,
            )

        raise gws.Error(f'API {rx.route=} invalid response type: {type(res)}')

    ##

    @route('POST api/v1/auth/logout')
    def on_post_auth_logout(self, rx: Request):
        """Log out, deleting the session.

        Args:
            rx: API request.
        """
        am = self.root.app.authMgr
        am.sessionMgr.delete(rx.sess)
        gws.log.debug(f'{self=} {rx=}')

    @route('GET api/v1/auth/providers')
    def on_get_auth_providers(self, rx: Request) -> list[api.AuthProvider]:
        """Return the authentication providers.

        Args:
            rx: API request.

        Returns:
            A single username and password provider.
        """
        return [
            api.AuthProvider(type='credentials', id='credentials', name='Username / Password'),
        ]

    @route('GET api/v1/server/info')
    def on_get_server_info(self, rx: Request) -> api.ServerInfo:
        """Return the server information.

        Args:
            rx: API request.

        Returns:
            Server information.
        """
        return api.ServerInfo(
            version=self.root.app.version,
            auth_providers=self.on_get_auth_providers(rx),
            signup_url='',
            whitelabel={},
        )

    @route('GET api/v1/status')
    def on_get_status(self, rx: Request) -> api.Status:
        """Return the server status.

        Args:
            rx: API request.

        Returns:
            Server status, always ``ok``.
        """
        return api.Status(
            version=self.root.app.version,
            database='ok',
            storage='ok',
            status_page_url=None,
            incident_message=None,
            incident_timestamp_utc=None,
            maintenance_message=None,
            maintenance_start_timestamp_utc=None,
            maintenance_end_timestamp_utc=None,
        )

    @route('GET api/v1/users/(?P<username>[^/]+)/organizations')
    def on_get_user_organizations(self, rx: Request) -> list:
        """Return the organizations of a user.

        Args:
            rx: API request.

        Returns:
            An empty list, organizations are not supported.
        """
        return []

    @route('GET api/v1/subscriptions/(?P<username>[^/]+)/current')
    def on_get_subscription(self, rx: Request) -> api.Subscription:
        """Return the subscription of a user.

        Args:
            rx: API request.

        Returns:
            A dummy active subscription.
        """
        return api.Subscription(
            plan_display_name='',
            active_storage_total_bytes=0,
            storage_used_bytes=0,
            plan_storage_threshold_warning_bytes=0,
            plan_storage_threshold_critical_bytes=0,
            status='active_paid',
        )

    @route('POST api/v1/auth/token')
    def on_post_auth_token(self, rx: Request) -> api.AuthToken:
        """Log in with username and password and create a session.

        Args:
            rx: API request.

        Returns:
            Session token and user information.

        Raises:
            ``gws.AuthenticationError``: If the credentials are invalid.
        """
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
        """Return the authenticated user.

        Args:
            rx: API request.

        Returns:
            User information.
        """
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
        """Return the QField projects the user can use.

        Args:
            rx: API request.

        Returns:
            Projects, paged by the ``limit`` and ``offset`` query parameters.
        """
        limit = int(rx.qs.get('limit', 100))
        offset = int(rx.qs.get('offset', 0))
        qps = self.get_qfc_projects(rx.user)
        return [_format_project(qp, rx) for qp in qps[offset : offset + limit]]

    @route('POST api/v1/projects')
    def on_post_projects(self, rx: Request):
        """Create a project. Not supported.

        Args:
            rx: API request.

        Raises:
            ``gws.BadRequestError``: Always.
        """
        raise gws.BadRequestError('creating projects is not supported')

    @route('GET api/v1/projects/(?P<project_id>[^/]+)')
    def on_get_projects_id(self, rx: Request) -> api.Project:
        """Return a QField project.

        Args:
            rx: API request.

        Returns:
            Project.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        self.set_qfc_project_from_parts(rx)
        return _format_project(rx.qfcProject, rx)

    @route('POST api/v1/jobs')
    def on_post_jobs(self, rx: Request) -> api.Job:
        """Create a job. Only ``package`` jobs are supported.

        Args:
            rx: API request.

        Returns:
            The scheduled job.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If the job type is not supported.
        """
        project_id = rx.post.get('project_id', '')
        type = rx.post.get('type', '')
        self.set_qfc_project(project_id, rx)
        if type != api.TypeEnum.package:
            raise gws.BadRequestError(f'unsupported job type: {type!r}')

        job = self.create_package_job(rx)
        return _format_job(job, rx)

    @route('GET api/v1/jobs')
    def on_get_jobs(self, rx: Request):
        """List jobs. Not supported.

        Args:
            rx: API request.

        Raises:
            ``gws.BadRequestError``: Always.
        """
        raise gws.BadRequestError('listing jobs is not supported')

    @route('GET api/v1/jobs/(?P<job_id>[^/]+)')
    def on_get_jobs_id(self, rx: Request) -> api.Job:
        """Return a job.

        Args:
            rx: API request.

        Returns:
            Job.

        Raises:
            ``gws.NotFoundError``: If the job is not found.
        """
        job_id = rx.parts.get('job_id', '')
        job = self.get_job(job_id, rx.user)
        if not job:
            raise gws.NotFoundError(f'Job {job_id!r} not found')
        return _format_job(job, rx)

    @route('GET api/v1/packages/(?P<project_id>[^/]+)/(?P<package_version>[^/]+)')
    def on_get_package(self, rx: Request) -> api.Package:
        """Return the latest package of a QField project.

        Args:
            rx: API request.

        Returns:
            Package with its files. The package version is ignored.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
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
        """Return a file from the latest package.

        Args:
            rx: API request.

        Returns:
            File content. A request with a ``Range`` header gets status 416, so that QField downloads the full file.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the file is not in the package.
        """
        self.set_qfc_project_from_parts(rx)

        file_name = rx.parts.get('file_name', '')
        path_map = self.get_latest_package_path_map(rx)
        for fname, p in path_map.items():
            if file_name == fname:
                # QField resumes interrupted downloads with "Range: bytes=N-" and appends the body to its partial file.
                # Ranges are not supported, so refuse with 416: QField then drops the partial file and downloads in full.
                if rx.req.header('Range'):
                    return gws.ContentResponse(
                        status=416,
                        content='',
                        headers={'Content-Range': f'bytes */{osx.file_size(p)}'},
                    )
                return gws.ContentResponse(contentPath=p)
        raise gws.NotFoundError(f'file {file_name!r} not found')

    @route('GET api/v1/files/(?P<project_id>[^/]+)')
    def on_get_files(self, rx: Request) -> list[api.PackageFile]:
        """Return the files of the latest package.

        Args:
            rx: API request.

        Returns:
            Package files.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        self.set_qfc_project_from_parts(rx)
        path_map = self.get_latest_package_path_map(rx)
        return _format_files(path_map)

    @route('GET api/v1/files/thumbnails/(?P<project_id>[^/]+)')
    def on_get_thumbnail(self, rx: Request) -> gws.ContentResponse:
        """Return the thumbnail of a QField project.

        Args:
            rx: API request.

        Returns:
            Thumbnail image.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the project has no thumbnail.
        """
        self.set_qfc_project_from_parts(rx)
        path = rx.qfcProject.thumbnail
        if not path or not gws.u.is_file(path):
            raise gws.NotFoundError(f'no thumbnail for {rx.qfcProject.uid!r}')
        return gws.ContentResponse(contentPath=path)

    @route('POST api/v1/deltas/(?P<project_id>[^/]+)')
    def on_post_deltas(self, rx: Request):
        """Store a delta payload and apply its changes.

        The payload is uploaded as a multipart file. After the patcher returns, all deltas of the payload are marked as applied.

        Args:
            rx: API request.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If the payload cannot be read.
        """
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
        """Return the stored deltas of a payload.

        Args:
            rx: API request.

        Returns:
            Stored deltas.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the payload is not found.
        """
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
        """Apply a file upload.

        Args:
            rx: API request.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If the upload cannot be read.
        """
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
        """Return a new packager. Override to use a custom packager.

        Returns:
            Packager.
        """
        return packager.Object()

    def get_patcher(self) -> patcher.Object:
        """Return a new patcher. Override to use a custom patcher.

        Returns:
            Patcher.
        """
        return patcher.Object()

    ##

    def get_caps(self, qfc_project: core.QfcProject) -> caps.Caps:
        """Return the capabilities of a QField project, from the cache if they are still valid.

        New capabilities are also written to ``caps.pickle`` in the project cache directory.

        Args:
            qfc_project: QField project.

        Returns:
            Capabilities.
        """
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
        """Return cached capabilities, if the QGIS project source has not changed.

        Args:
            qfc_project: QField project.

        Returns:
            Capabilities, or ``None`` if not cached or outdated.
        """
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
        """Authenticate a user and create a session.

        Sets ``sess``, ``user`` and ``token`` in the request.

        Args:
            credentials: Data with ``username`` and ``password``.
            rx: API request.

        Raises:
            ``gws.AuthenticationError``: If the credentials are invalid.
        """
        am = self.root.app.authMgr
        user = am.authenticate(self.method, credentials, rx.req)
        if not user:
            raise gws.AuthenticationError('invalid username or password')
        rx.sess = am.sessionMgr.create(self.method, user)
        rx.user = user
        rx.token = rx.sess.uid

    def authorize_from_token(self, rx: Request):
        """Authorize a request by the ``Authorization: Token ...`` header.

        Sets ``sess``, ``user`` and ``token`` in the request and touches the session.

        Args:
            rx: API request.

        Raises:
            ``gws.AuthenticationError``: If the header is missing, or the token is invalid or belongs to another method.
            ``gws.ForbiddenError``: If the method cannot be used in this context, e.g. without a secure connection.
        """
        h = rx.req.header('Authorization', '')
        m = re.match(r'^Token (.+)$', h)
        if not m:
            raise gws.AuthenticationError('token_auth: missing or invalid Authorization header')
        token = m.group(1)
        am = self.root.app.authMgr
        if not am.can_use_method(rx.req, self.method):
            raise gws.ForbiddenError('token_auth: insecure_context')
        sess = am.sessionMgr.get(token)
        if not sess:
            raise gws.AuthenticationError('token_auth: invalid or expired token')
        if not sess.method or sess.method.uid != self.method.uid:
            raise gws.AuthenticationError(f'token_auth: wrong method {sess.method=}')
        rx.sess = sess
        rx.user = sess.user
        rx.token = sess.uid
        am.sessionMgr.touch(sess)
        gws.log.debug(f'token_auth: ok: {rx.token=} {rx.user.uid=} {rx.user.loginName=}')

    ##

    def set_qfc_project(self, uid: str, rx: Request):
        """Set the QField project of a request.

        Args:
            uid: QField project uid.
            rx: API request.

        Raises:
            ``gws.NotFoundError``: If the project is not found or the user cannot use it.
        """
        qp = self.get_qfc_project(uid, rx.user)
        if not qp:
            raise gws.NotFoundError(f'project {uid!r} not found')
        rx.qfcProject = qp

    def set_qfc_project_from_parts(self, rx: Request):
        """Set the QField project of a request from the ``project_id`` path variable.

        Args:
            rx: API request.

        Raises:
            ``gws.NotFoundError``: If the project is not found or the user cannot use it.
        """
        uid = rx.parts.get('project_id', '')
        self.set_qfc_project(uid, rx)

    ##

    def get_qfc_projects(self, user: gws.User) -> list[core.QfcProject]:
        """Return the QField projects a user can use.

        Args:
            user: User.

        Returns:
            QField projects.
        """
        return [p for p in self.qfcProjects if user.can_use(p)]

    def get_qfc_project(self, qfc_project_uid: str, user: gws.User) -> Optional[core.QfcProject]:
        """Return a QField project if the user can use it.

        Args:
            qfc_project_uid: QField project uid.
            user: User.

        Returns:
            QField project, or ``None`` if not found.
        """
        for qp in self.get_qfc_projects(user):
            if qp.uid == qfc_project_uid:
                return qp

    ##

    def create_package_job(self, rx: Request) -> gws.Job:
        """Create and schedule a packaging job for the QField project of a request.

        Args:
            rx: API request.

        Returns:
            The scheduled job.
        """
        mgr = self.root.app.jobMgr
        p = WorkerPayload(
            actionUid=self.uid,
            jobType='package',
            qfcProjectUid=rx.qfcProject.uid,
            projectUid=rx.project.uid if rx.project else None,
        )
        job = mgr.create_job(
            PackageWorker,
            rx.user,
            payload=gws.u.to_dict(p),
        )
        return mgr.schedule_job(job)

    def create_package_from_worker(self, worker: 'PackageWorker', pa: WorkerPayload):
        """Create a package in a new package directory. Called by the package worker.

        Packages older than an hour are removed first.

        Args:
            worker: Package worker.
            pa: Job payload.
        """
        project = worker.user.require_project(pa.projectUid) if pa.projectUid else None
        qfc_project = gws.u.require(self.get_qfc_project(pa.qfcProjectUid, worker.user))

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

    def create_package_from_cli(self, qfc_project_uid: str, target_dir: str, project: Optional[gws.Project], user: gws.User):
        """Create a package in a given directory.

        Args:
            qfc_project_uid: QField project uid.
            target_dir: Directory to write the package into.
            project: GWS project context.
            user: User the package is created for.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        qfc_project = self.get_qfc_project(qfc_project_uid, user)
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

    def get_job(self, job_id: str, user: gws.User) -> Optional[gws.Job]:
        """Return a job of a user.

        Args:
            job_id: Job uid.
            user: User.

        Returns:
            The job, or ``None`` if not found.
        """
        return self.root.app.jobMgr.get_job(job_id, user=user)

    ##

    def store_delta_payload(self, payload: api.DeltasPayload, rx: Request):
        """Store the deltas of a payload with the pending status.

        Stored payloads older than an hour are removed first.

        Args:
            payload: Delta payload.
            rx: API request.
        """
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
        """Mark all stored deltas of a payload as applied.

        Args:
            payload_id: Payload id.
            rx: API request.
        """
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
        """Load the stored deltas of a payload.

        Args:
            payload_id: Payload id.
            rx: API request.

        Returns:
            Stored deltas, or ``None`` if not found, unreadable or stored by another user.
        """
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
        """Return the base directory of a QField project, creating it if needed.

        The directory is ``<VAR_DIR>/qfieldcloud/projects/<uid>``. Packages, caches
        and stored deltas are kept below it.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path.
        """
        return gws.u.ensure_dir(f'{gws.c.VAR_DIR}/qfieldcloud/projects/{qfc_project.uid}')

    def fs_latest_package_dir(self, qfc_project: core.QfcProject) -> Optional[str]:
        """Return the directory of the latest complete package.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path, or ``None`` if there is no complete package.
        """
        base_dir = self.fs_project_base_dir(qfc_project)
        for pkg in sorted(osx.find_directories(base_dir, deep=False), reverse=True):
            m = re.search(r'package_(\d+)', pkg)
            if m and gws.u.is_file(f'{pkg}/{packager.COMPLETE_FILE}'):
                return pkg

    def fs_new_package_dir(self, qfc_project: core.QfcProject, uid: str) -> str:
        """Create a directory for a new package.

        Args:
            qfc_project: QField project.
            uid: Package uid.

        Returns:
            Directory path.
        """
        base_dir = self.fs_project_base_dir(qfc_project)
        pkg_dir = gws.u.ensure_dir(f'{base_dir}/package_{uid}')
        return pkg_dir

    def fs_cleanup_old_packages(self, qfc_project: core.QfcProject, keep_seconds: int = 3600):
        """Remove package directories older than ``keep_seconds``.

        Args:
            qfc_project: QField project.
            keep_seconds: Maximum age in seconds.
        """
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
        """Return the cache directory of a QField project, creating it if needed.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path.
        """
        base_dir = self.fs_project_base_dir(qfc_project)
        return gws.u.ensure_dir(f'{base_dir}/cache')

    def fs_project_deltas_dir(self, qfc_project: core.QfcProject) -> str:
        """Return the deltas directory of a QField project, creating it if needed.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path.
        """
        base_dir = self.fs_project_base_dir(qfc_project)
        return gws.u.ensure_dir(f'{base_dir}/deltas')

    def fs_delta_payload_path(self, qfc_project: core.QfcProject, payload_id: str) -> str:
        """Return the path of a stored delta payload.

        Args:
            qfc_project: QField project.
            payload_id: Payload id.

        Returns:
            File path.
        """
        d = self.fs_project_deltas_dir(qfc_project)
        u = gws.u.to_uid(payload_id)
        return f'{d}/{u}.json'

    def fs_cleanup_old_deltas(self, qfc_project: core.QfcProject, keep_seconds: int = 3600):
        """Remove stored delta payloads older than ``keep_seconds``.

        Args:
            qfc_project: QField project.
            keep_seconds: Maximum age in seconds.
        """
        d = self.fs_project_deltas_dir(qfc_project)
        now = dtx.now().timestamp()
        for f in osx.find_files(d, deep=False):
            t = osx.file_mtime(f)
            if now - t > keep_seconds:
                gws.log.info(f'fs_cleanup_old_deltas: removing old delta: {f=}')
                osx.unlink(f)

    def get_latest_package_path_map(self, rx: Request) -> dict[str, str]:
        """Return the path map of the latest package of the request's QField project.

        Args:
            rx: API request.

        Returns:
            Paths on disk by package file name, empty if there is no package.
        """
        d = self.fs_latest_package_dir(rx.qfcProject)
        try:
            return gws.lib.jsonx.from_path(f'{d}/{packager.PATH_MAP_FILE}')
        except Exception:
            return {}


##


class PackageWorker(gws.base.job.worker.Object):
    """Job worker that creates a package."""

    @classmethod
    def run(cls, root: gws.Root, job: gws.Job):
        """Run a packaging job.

        Args:
            root: Root object.
            job: The job.
        """
        w = cls(root, job.user, job)
        w.work()

    def work(self):
        """Create the package and mark the job as complete."""
        self.update_job(state=gws.JobState.running)
        pa = WorkerPayload(gws.u.require(self.get_job()).payload)
        action = cast(Object, self.root.get(pa.actionUid))
        action.create_package_from_worker(self, pa)
        self.update_job(state=gws.JobState.complete)


##


_DATE_CREATED = '2025-10-10T14:00:00'


def _error_response(status: int, code: str, exc: Exception) -> gws.ContentResponse:
    """Log an error and return a JSON error response."""
    gws.log.warning(f'qfieldcloudApi: {status} {code} cause={exc!r}')
    return gws.ContentResponse(
        status=status,
        content=gws.lib.jsonx.to_string({'code': code}),
        mimeType=gws.lib.mime.JSON,
    )


def _format_project(qp: core.QfcProject, rx: Request) -> api.Project:
    """Convert a QField project to an API project."""
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
    """Convert a package path map to a list of API package files."""
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
    """Convert a GWS job to an API job."""
    status_map = {
        gws.JobState.open: api.JobStatusEnum.pending,
        gws.JobState.running: api.JobStatusEnum.started,
        gws.JobState.complete: api.JobStatusEnum.finished,
        gws.JobState.error: api.JobStatusEnum.failed,
    }

    return api.Job(
        id=job.uid,
        type=job.payload.get('jobType', ''),
        created_at=dtx.to_iso_string(job.timeCreated),
        created_by=1,
        project_id=job.payload.get('qfcProjectUid', ''),
        status=status_map.get(job.state, api.JobStatusEnum.pending),
        updated_at=dtx.to_iso_string(job.timeUpdated),
        started_at=dtx.to_iso_string(job.timeUpdated) if job.state == gws.JobState.running else None,
        finished_at=dtx.to_iso_string(job.timeUpdated) if job.state == gws.JobState.complete else None,
    )


def _get_sha256(path: str) -> str:
    """Return the SHA-256 hash of a file."""
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def _get_time_iso(path: str) -> str:
    """Return the modification time of a file as an ISO string."""
    t = osx.file_mtime(path)
    return dtx.to_iso_string(dtx.from_timestamp(t))


def _get_md5sum(path: str) -> str:
    """Return the MD5 hash of a file, as computed by QField."""
    with open(path, 'rb') as f:
        return _get_md5sum_file(f)


def _get_md5sum_file(fp, part_size: int = 8 * 1024 * 1024) -> str:
    """Return a plain MD5 for small files, or an S3-style multipart ETag for larger ones, as QField's ``fileEtag``."""
    fp.seek(0, 2)
    file_size = fp.tell()
    fp.seek(0)

    if file_size <= part_size:
        hash = hashlib.md5()
        hash.update(fp.read())
        return hash.hexdigest()

    md5_sums = b''
    read_size = 0

    while read_size < file_size:
        hash = hashlib.md5()
        hash.update(fp.read(part_size))
        md5_sums += hash.digest()
        read_size += part_size

    hash = hashlib.md5()
    hash.update(md5_sums)
    return f'{hash.hexdigest()}-{read_size // part_size}'

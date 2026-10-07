"""QField Cloud API request handler."""

from typing import Optional, cast

import re
import hashlib

import gws
import gws.lib.mime
import gws.lib.jsonx
import gws.lib.sqlitex
import gws.lib.datetimex as dtx
import gws.lib.osx as osx

from . import core, patcher, api, action_base, delta, auth


def route(pattern: str):
    """Decorator that marks a method as an API route handler.

    Args:
        pattern: Regular expression matched against ``"<METHOD> <path>"``. Named groups are passed to the handler in ``Handler.parts``.

    Returns:
        The decorator.
    """

    def decorator(fn):
        fn._route_pattern = pattern
        return fn

    return decorator


class Handler:
    """Handler for a single QFieldCloud API request.

    Finds the route for the request, authenticates the user and calls the route method.
    The handler holds the state of the request only, persistent data is accessed via ``action``.
    """

    action: action_base.BaseAction
    """The action that received the request."""
    req: gws.WebRequester
    """The original web request."""
    params: gws.Request
    """Command request."""
    apiRoute: str
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
    auth: auth.Result
    """Authentication result: user, session and token."""

    def __init__(self, action: action_base.BaseAction):
        """Create a handler.

        Args:
            action: The action that received the request.
        """
        self.action = action
        self.apiRoute = ''
        self.parts = {}
        self.qs = {}
        self.post = {}
        self.project = None

    def handle(self, req: gws.WebRequester, params: gws.Request) -> gws.ContentResponse:
        """Find the route method for a request and call it.

        The path can start with ``projectUid/<uid>`` to set the GWS project.
        If the action belongs to a project, that project is used instead.

        Args:
            req: Web requester.
            params: Command request.

        Returns:
            The response.

        Raises:
            ``gws.NotFoundError``: If the path is empty, the GWS project is not found or no route matches.
            ``gws.ForbiddenError``: If the user cannot use the GWS project.
        """
        self.req = req
        self.params = params

        path = self.req.path().strip('/')
        if not path:
            raise gws.NotFoundError('API path not specified')

        path_parts = path.split('/')
        uid = self.params.get('projectUid')

        if path_parts[0] == 'projectUid':
            if len(path_parts) < 2:
                raise gws.NotFoundError('gws project UID not specified')
            uid = path_parts[1]
            path = '/'.join(path_parts[2:])

        project = cast(Optional[gws.Project], self.action.find_closest(gws.ext.object.project))
        project_uid = project.uid if project else uid
        if project_uid:
            project = self.req.user.require_project(project_uid)

        path = path.strip('/')
        route = f'{self.req.method} {path}'

        for name in dir(self):
            fn = getattr(self, name)
            if callable(fn) and hasattr(fn, '_route_pattern'):
                m = re.match(f'^{fn._route_pattern}$', route)
                if m:
                    self.project = project
                    self.apiRoute = route
                    self.parts = m.groupdict()
                    self.qs = self.req.query_params()
                    return self._handle_route(fn)

        raise gws.NotFoundError(f'API {route=} not found')

    _public_routes = (
        'GET api/v1/auth/providers',
        'POST api/v1/auth/token',
        'GET api/v1/server/info',
        'GET api/v1/status',
    )

    def _handle_route(self, fn) -> gws.ContentResponse:
        """Read the request payload, authenticate non-public routes, call the route method and convert its result to a response."""
        if self.req.isApi:
            self.post = self.req.struct()
        elif self.req.isForm:
            self.post = dict(self.req.form())
        elif self.req.isPost:
            self.post = dict(raw=self.req.data())

        gws.log.debug(f'API_REQUEST {self.apiRoute=} -> {fn.__name__} {self.parts=} {self.qs=}')

        if self.apiRoute not in self._public_routes:
            self.auth = self.action.authMethod.authenticate_from_token(self.req)

        res = fn()

        if res is None:
            return gws.ContentResponse(content='')

        if isinstance(res, gws.ContentResponse):
            return res

        if isinstance(res, (list, dict, gws.Data)):
            return gws.ContentResponse(
                content=gws.lib.jsonx.to_string(res),
                mimeType=gws.lib.mime.JSON,
            )

        raise gws.Error(f'API {self.apiRoute=} invalid response type: {type(res)}')

    ##

    @route('POST api/v1/auth/logout')
    def on_post_auth_logout(self):
        """Log out, deleting the session."""
        am = self.action.root.app.authMgr
        am.sessionMgr.delete(self.auth.sess)
        gws.log.debug(f'logout: {self.auth.user.loginName=}')

    @route('GET api/v1/auth/providers')
    def on_get_auth_providers(self) -> list[api.AuthProvider]:
        """Return the authentication providers.

        Returns:
            A single username and password provider.
        """
        return [
            api.AuthProvider(type='credentials', id='credentials', name='Username / Password'),
        ]

    @route('GET api/v1/server/info')
    def on_get_server_info(self) -> api.ServerInfo:
        """Return the server information.

        Returns:
            Server information.
        """
        return api.ServerInfo(
            version=self.action.root.app.version,
            auth_providers=self.on_get_auth_providers(),
            signup_url='',
            whitelabel={},
        )

    @route('GET api/v1/status')
    def on_get_status(self) -> api.Status:
        """Return the server status.

        Returns:
            Server status, always ``ok``.
        """
        return api.Status(
            version=self.action.root.app.version,
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
    def on_get_user_organizations(self) -> list:
        """Return the organizations of a user.

        Returns:
            An empty list, organizations are not supported.
        """
        return []

    @route('GET api/v1/subscriptions/(?P<username>[^/]+)/current')
    def on_get_subscription(self) -> api.Subscription:
        """Return the subscription of a user.

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
    def on_post_auth_token(self) -> api.AuthToken:
        """Log in with username and password and create a session.

        Returns:
            Session token and user information.

        Raises:
            ``gws.AuthenticationError``: If the credentials are invalid.
        """
        self.auth = self.action.authMethod.authenticate_from_credentials(
            gws.Data(
                username=self.post.get('username', ''),
                password=self.post.get('password', ''),
            ),
            self.req,
        )
        return api.AuthToken(
            token=self.auth.token,
            expires_at=dtx.to_iso_string(self.auth.expiresAt),
            username=self.auth.user.loginName,
            type=api.UserType.person,
            full_name=self.auth.user.displayName,
            avatar_url='',
            email='',
            first_name='',
            last_name='',
        )

    @route('GET api/v1/auth/user')
    def on_get_auth_user(self) -> api.CompleteUser:
        """Return the authenticated user.

        Returns:
            User information.
        """
        return api.CompleteUser(
            username=self.auth.user.loginName,
            type=api.UserType.person,
            full_name=self.auth.user.displayName,
            avatar_url='',
            email='',
            first_name='',
            last_name='',
        )

    @route('GET api/v1/projects')
    def on_get_projects(self) -> list[api.Project]:
        """Return the QField projects the user can use.

        Returns:
            Projects, paged by the ``limit`` and ``offset`` query parameters.
        """
        limit = int(self.qs.get('limit', 100))
        offset = int(self.qs.get('offset', 0))
        qps = self.action.get_qfc_projects(self.auth.user)
        return [_format_project(qp, self.auth.user) for qp in qps[offset : offset + limit]]

    @route('POST api/v1/projects')
    def on_post_projects(self):
        """Create a project. Not supported.

        Raises:
            ``gws.BadRequestError``: Always.
        """
        raise gws.BadRequestError('creating projects is not supported')

    @route('GET api/v1/projects/(?P<project_id>[^/]+)')
    def on_get_projects_id(self) -> api.Project:
        """Return a QField project.

        Returns:
            Project.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        self.set_qfc_project_from_parts()
        return _format_project(self.qfcProject, self.auth.user)

    @route('POST api/v1/jobs')
    def on_post_jobs(self) -> api.Job:
        """Create a job. Only ``package`` jobs are supported.

        Returns:
            The scheduled job.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If the job type is not supported.
        """
        project_id = self.post.get('project_id', '')
        type = self.post.get('type', '')
        self.set_qfc_project(project_id)
        if type != api.TypeEnum.package:
            raise gws.BadRequestError(f'unsupported job type: {type!r}')

        job = self.create_package_job()
        return _format_job(job)

    @route('GET api/v1/jobs')
    def on_get_jobs(self):
        """List jobs. Not supported.

        Raises:
            ``gws.BadRequestError``: Always.
        """
        raise gws.BadRequestError('listing jobs is not supported')

    @route('GET api/v1/jobs/(?P<job_id>[^/]+)')
    def on_get_jobs_id(self) -> api.Job:
        """Return a job.

        Returns:
            Job.

        Raises:
            ``gws.NotFoundError``: If the job is not found.
        """
        job_id = self.parts.get('job_id', '')
        job = self.action.get_job(job_id, self.auth.user)
        if not job:
            raise gws.NotFoundError(f'Job {job_id!r} not found')
        return _format_job(job)

    @route('GET api/v1/packages/(?P<project_id>[^/]+)/(?P<package_version>[^/]+)')
    def on_get_package(self) -> api.Package:
        """Return the latest package of a QField project.

        Returns:
            Package with its files. The package version is ignored.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        self.set_qfc_project_from_parts()

        # @TODO do we need versions?
        # @TODO do we need layers?
        # package_version = self.parts.get('package_version', '')

        path_map = self.action.latest_package_path_map(self.qfcProject)
        return api.Package(
            files=_format_files(path_map),
            layers=[],
            status=api.JobStatusEnum.finished,
            package_id=self.qfcProject.uid,
            packaged_at=dtx.to_iso_string(),
            data_last_updated_at=dtx.to_iso_string(),
        )

    @route('GET api/v1/packages/(?P<project_id>[^/]+)/(?P<package_version>[^/]+)/files/(?P<file_name>.+)')
    def on_get_package_file(self) -> gws.ContentResponse:
        """Return a file from the latest package.

        Returns:
            File content. A request with a ``Range`` header gets status 416, so that QField downloads the full file.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the file is not in the package.
        """
        self.set_qfc_project_from_parts()

        file_name = self.parts.get('file_name', '')
        path_map = self.action.latest_package_path_map(self.qfcProject)
        for fname, p in path_map.items():
            if file_name == fname:
                # QField resumes interrupted downloads with "Range: bytes=N-" and appends the body to its partial file.
                # Ranges are not supported, so refuse with 416: QField then drops the partial file and downloads in full.
                if self.req.header('Range'):
                    return gws.ContentResponse(
                        status=416,
                        content='',
                        headers={'Content-Range': f'bytes */{osx.file_size(p)}'},
                    )
                return gws.ContentResponse(contentPath=p)
        raise gws.NotFoundError(f'file {file_name!r} not found')

    @route('GET api/v1/files/(?P<project_id>[^/]+)')
    def on_get_files(self) -> list[api.PackageFile]:
        """Return the files of the latest package.

        Returns:
            Package files.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
        """
        self.set_qfc_project_from_parts()
        path_map = self.action.latest_package_path_map(self.qfcProject)
        return _format_files(path_map)

    @route('GET api/v1/files/thumbnails/(?P<project_id>[^/]+)')
    def on_get_thumbnail(self) -> gws.ContentResponse:
        """Return the thumbnail of a QField project.

        Returns:
            Thumbnail image.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the project has no thumbnail.
        """
        self.set_qfc_project_from_parts()
        path = self.qfcProject.thumbnail
        if not path or not gws.u.is_file(path):
            raise gws.NotFoundError(f'no thumbnail for {self.qfcProject.uid!r}')
        return gws.ContentResponse(contentPath=path)

    @route('POST api/v1/deltas/(?P<project_id>[^/]+)')
    def on_post_deltas(self):
        """Store a delta payload and apply its changes.

        The payload is uploaded as a multipart file. Deltas are identified by their content,
        so a delta pushed again unchanged, also under another payload id, is not applied again.
        The deltas that are new, or failed before, or have been pending for longer than
        ``delta.PENDING_TIMEOUT``, are applied all together or not at all.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If the payload cannot be read.
            ``core.DeltaApplyError``: If a delta is still being applied, or the deltas cannot be applied.
        """
        self.set_qfc_project_from_parts()

        try:
            text = self.post['file'].stream.read().decode('utf-8')
            payload = delta.parse_payload(text)
        except Exception as exc:
            raise gws.BadRequestError(f'invalid delta file: {exc}')

        user_name = self.auth.user.loginName
        delta.cleanup(self.delta_db(), self.action.deltaLifeTime)

        uids = []
        drs = []

        for d in payload.deltas:
            uid = gws.u.sha256([user_name, d])
            uids.append(uid)
            dr = delta.get_one(self.delta_db(), uid, user_name)
            if dr:
                ds = delta.status(dr, payload.id)
                if ds.status == api.DeltaStatusType.applied:
                    continue
                if ds.status == api.DeltaStatusType.pending:
                    raise core.DeltaApplyError(f'changes {payload.id!r} are being applied, try again later')
            drs.append(delta.new_record(uid, d, user_name))

        delta.link_payload(self.delta_db(), payload.id, uids, user_name)
        delta.store_all(self.delta_db(), drs)

        if not drs:
            gws.log.info(f'deltas {payload.id!r}: already applied')
            return

        try:
            args = patcher.Args(
                qfcProject=self.qfcProject,
                caps=self.action.get_caps(self.qfcProject),
                db=self.action.get_db_provider(self.qfcProject),
                project=self.project,
                user=self.auth.user,
                baseDir='',
                changes=delta.extract_changes(drs),
            )
            self.action.get_patcher().apply_changes(self.action.root, args)
        except Exception as exc:
            gws.log.exception(f'deltas {payload.id!r}: failed to apply')
            delta.set_status_for_all(self.delta_db(), drs, api.DeltaStatusType.error, str(exc))
            raise core.DeltaApplyError(f'changes {payload.id!r} could not be applied, see the server log') from exc

        delta.set_status_for_all(self.delta_db(), drs, api.DeltaStatusType.applied)

    @route('GET api/v1/deltas/(?P<project_id>[^/]+)')
    def on_get_deltas(self) -> gws.ContentResponse:
        """Return the deltas the user has sent for a QField project, newest first.

        The list is paged by the ``limit`` and ``offset`` query parameters. If there are more deltas,
        the response has an ``X-Next-Page`` header. The ``ordering`` parameter is ignored.

        Returns:
            Delta statuses as JSON.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.BadRequestError``: If ``limit`` or ``offset`` is not a number.
        """
        self.set_qfc_project_from_parts()

        try:
            limit = int(self.qs.get('limit', 50))
            offset = int(self.qs.get('offset', 0))
        except ValueError as exc:
            raise gws.BadRequestError(f'invalid paging parameters: {exc}')

        rs = delta.get_history(self.delta_db(), self.auth.user.loginName, limit + 1, offset)
        dss = [delta.status(dr, payload_id) for dr, payload_id in rs[:limit]]

        headers = {}
        if len(rs) > limit:
            headers['X-Next-Page'] = str(offset + limit)

        return gws.ContentResponse(
            content=gws.lib.jsonx.to_string(dss),
            mimeType=gws.lib.mime.JSON,
            headers=headers,
        )

    @route('GET api/v1/deltas/(?P<project_id>[^/]+)/(?P<payload_id>.+)')
    def on_get_deltas_in_payload(self) -> list[core.DeltaStatus]:
        """Return the status of the deltas of a payload.

        Returns:
            Delta statuses.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If the payload is not found.
        """
        self.set_qfc_project_from_parts()

        payload_id = self.parts.get('payload_id', '')
        drs = delta.get_for_payload(self.delta_db(), payload_id, self.auth.user.loginName)
        dss = [delta.status(dr, payload_id) for dr in drs]
        if not dss:
            raise gws.NotFoundError(f'delta {payload_id=} not found')
        return dss

    @route('POST api/v1/files/(?P<project_id>[^/]+)/(?P<path>.+)')
    def on_post_file(self) -> gws.ContentResponse:
        """Apply a file upload.

        Returns:
            An empty response with status 201.

        Raises:
            ``gws.NotFoundError``: If the QField project is not found.
            ``gws.NotFoundError``: If no feature refers to the file.
            ``gws.BadRequestError``: If the upload cannot be read.
        """
        self.set_qfc_project_from_parts()

        path = self.parts.get('path', '')
        try:
            fc = self.post['file'].stream.read()
        except Exception as exc:
            raise gws.BadRequestError(f'invalid file upload: {exc}')

        args = patcher.Args(
            qfcProject=self.qfcProject,
            caps=self.action.get_caps(self.qfcProject),
            project=self.project,
            user=self.auth.user,
            baseDir='',
            filePath=path,
            fileContent=fc,
        )
        if not self.action.get_patcher().apply_upload(self.action.root, args):
            raise gws.NotFoundError(f'no feature found for {path!r}')
        return gws.ContentResponse(status=201, content='')

    ##

    ##

    def set_qfc_project(self, uid: str):
        """Set the QField project of the request.

        Args:
            uid: QField project uid.

        Raises:
            ``gws.NotFoundError``: If the project is not found or the user cannot use it.
        """
        qp = self.action.get_qfc_project(uid, self.auth.user)
        if not qp:
            raise gws.NotFoundError(f'project {uid!r} not found')
        self.qfcProject = qp

    def set_qfc_project_from_parts(self):
        """Set the QField project of the request from the ``project_id`` path variable.

        Raises:
            ``gws.NotFoundError``: If the project is not found or the user cannot use it.
        """
        uid = self.parts.get('project_id', '')
        self.set_qfc_project(uid)

    ##

    def create_package_job(self) -> gws.Job:
        """Create and schedule a packaging job for the QField project of the request.

        Returns:
            The scheduled job.
        """
        mgr = self.action.root.app.jobMgr
        p = action_base.WorkerPayload(
            actionUid=self.action.uid,
            jobType='package',
            qfcProjectUid=self.qfcProject.uid,
            projectUid=self.project.uid if self.project else None,
        )
        job = mgr.create_job(
            action_base.PackageWorker,
            self.auth.user,
            payload=gws.u.to_dict(p),
        )
        return mgr.schedule_job(job)

    def delta_db(self) -> gws.lib.sqlitex.Object:
        """Return the delta database of the request's QField project.

        Returns:
            Database wrapper.
        """

        return gws.lib.sqlitex.Object(
            self.action.deltas_db_path(self.qfcProject),
            init_ddl=delta.STORAGE_DDL,
        )


##

_DATE_CREATED = '2025-10-10T14:00:00'


def _format_project(qp: core.QfcProject, user: gws.User) -> api.Project:
    """Convert a QField project to an API project."""
    return api.Project(
        id=qp.uid,
        name=qp.title,
        owner=user.loginName,
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


def _format_job(job: gws.Job) -> api.Job:
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

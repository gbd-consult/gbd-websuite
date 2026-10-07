"""QField Cloud API action: projects, capabilities, packaging and directories."""

from typing import Optional, cast

import re

import gws
import gws.base.job
import gws.base.action
import gws.lib.jsonx
import gws.lib.datetimex as dtx
import gws.lib.osx as osx

from . import core, packager, patcher, caps, auth


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


class BaseAction(gws.base.action.Object):
    """Base of the QField Cloud API action.

    Holds the configured QField projects and the authorization method, caches
    the project capabilities, creates packages and manages the project directories.
    Request handling is done by ``action_handler.Handler``.
    """

    qfcProjects: list[core.QfcProject]
    """Configured QField projects."""
    capsCache: dict[str, caps.Caps]
    """Capabilities by QField project uid. Not serialized."""
    authMethod: auth.Object
    """Authorization method for QField clients."""
    deltaLifeTime: int
    """How long stored delta payloads are kept, in seconds."""

    def configure(self):
        self.deltaLifeTime = self.cfg('deltaLifeTime', default=3600 * 24 * 30)
        self.authMethod = cast(
            auth.Object,
            self.create_child(gws.ext.object.authMethod, self.cfg('auth'), type='qfieldcloud'),
        )
        self.root.app.authMgr.add_method(self.authMethod)
        self.qfcProjects = []
        for p in self.cfg('projects') or []:
            qp = self.create_child(core.QfcProject, p)
            if qp:
                self.qfcProjects.append(cast(core.QfcProject, qp))

    def __getstate__(self):
        """Return the state for pickling, without the caps cache."""
        return gws.u.omit(vars(self), 'capsCache')

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
        pa.create_models()
        pa.assign_path_props()

        self.capsCache[qfc_project.uid] = pa.caps
        gws.log.debug(f'get_caps: {qfc_project.uid=}: created')
        return pa.caps

    def get_db_provider(self, qfc_project: core.QfcProject) -> gws.DatabaseProvider:
        """Return the database provider of the models of a QField project.

        Args:
            qfc_project: QField project.

        Returns:
            Database provider.

        Raises:
            ``gws.Error``: If the project has no models, or its models use more than one database provider.
        """
        cs = self.get_caps(qfc_project)

        db = None
        uids = set()
        for me in cs.modelMap.values():
            db = me.model.db
            uids.add(me.model.db.uid)

        if len(uids) != 1 or db is None:
            raise gws.Error(f'QField project {qfc_project.uid!r}: expected one database provider, found {sorted(uids)}')

        return db

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

    def create_package_from_worker(self, worker: 'PackageWorker', pa: WorkerPayload):
        """Create a package in a new package directory. Called by the package worker.

        Packages older than an hour are removed first.

        Args:
            worker: Package worker.
            pa: Job payload.
        """
        project = worker.user.require_project(pa.projectUid) if pa.projectUid else None
        qfc_project = gws.u.require(self.get_qfc_project(pa.qfcProjectUid, worker.user))

        self.cleanup_old_packages(qfc_project)

        uid = dtx.to_basic_string(with_ms=True)
        pkg_dir = self.new_package_dir(qfc_project, uid)
        args = packager.Args(
            uid=uid,
            qfcProject=qfc_project,
            caps=self.get_caps(qfc_project),
            project=project,
            user=worker.user,
            packageDir=pkg_dir,
            mapCacheDir=self.project_cache_dir(qfc_project),
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
            mapCacheDir=self.project_cache_dir(qfc_project),
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

    def project_base_dir(self, qfc_project: core.QfcProject) -> str:
        """Return the base directory of a QField project, creating it if needed.

        The directory is ``<VAR_DIR>/qfieldcloud/projects/<uid>``. Packages, caches
        and stored deltas are kept below it.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path.
        """
        return gws.u.ensure_dir(f'{gws.c.VAR_DIR}/qfieldcloud/projects/{qfc_project.uid}')

    def latest_package_dir(self, qfc_project: core.QfcProject) -> Optional[str]:
        """Return the directory of the latest complete package.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path, or ``None`` if there is no complete package.
        """
        base_dir = self.project_base_dir(qfc_project)
        for pkg in sorted(osx.find_directories(base_dir, deep=False), reverse=True):
            m = re.search(r'package_(\d+)', pkg)
            if m and gws.u.is_file(f'{pkg}/{packager.COMPLETE_FILE}'):
                return pkg

    def new_package_dir(self, qfc_project: core.QfcProject, uid: str) -> str:
        """Create a directory for a new package.

        Args:
            qfc_project: QField project.
            uid: Package uid.

        Returns:
            Directory path.
        """
        base_dir = self.project_base_dir(qfc_project)
        pkg_dir = gws.u.ensure_dir(f'{base_dir}/package_{uid}')
        return pkg_dir

    def cleanup_old_packages(self, qfc_project: core.QfcProject, keep_seconds: int = 3600):
        """Remove package directories older than ``keep_seconds``.

        Args:
            qfc_project: QField project.
            keep_seconds: Maximum age in seconds.
        """
        base_dir = self.project_base_dir(qfc_project)
        now = dtx.now().timestamp()
        for pkg in osx.find_directories(base_dir, deep=False):
            m = re.search(r'package_(\d+)', pkg)
            if not m:
                continue
            t = osx.file_mtime(pkg)
            if now - t > keep_seconds:
                gws.log.info(f'cleanup_old_packages: removing old package: {pkg=}')
                osx.rmdir(pkg)

    def latest_package_path_map(self, qfc_project: core.QfcProject) -> dict[str, str]:
        """Return the path map of the latest package of a QField project.

        Args:
            qfc_project: QField project.

        Returns:
            Paths on disk by package file name, empty if there is no package.
        """
        d = self.latest_package_dir(qfc_project)
        try:
            return gws.lib.jsonx.from_path(f'{d}/{packager.PATH_MAP_FILE}')
        except Exception:
            return {}

    def project_cache_dir(self, qfc_project: core.QfcProject) -> str:
        """Return the cache directory of a QField project, creating it if needed.

        Args:
            qfc_project: QField project.

        Returns:
            Directory path.
        """
        base_dir = self.project_base_dir(qfc_project)
        return gws.u.ensure_dir(f'{base_dir}/cache')

    def deltas_db_path(self, qfc_project: core.QfcProject) -> str:
        """Return the path of the delta database of a QField project.

        Args:
            qfc_project: QField project.

        Returns:
            File path.
        """
        base_dir = self.project_base_dir(qfc_project)
        return f'{base_dir}/deltas.sqlite'


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
        action = cast(BaseAction, self.root.get(pa.actionUid))
        action.create_package_from_worker(self, pa)
        self.update_job(state=gws.JobState.complete)

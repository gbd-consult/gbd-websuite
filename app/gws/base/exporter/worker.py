"""Export job worker."""

from typing import Optional, cast

import gws
import gws.base.job


class Object(gws.base.job.worker.Object):
    """Export job worker.

    Loads the requested features, runs the exporter and stores the export
    result in the job. Progress is reported in percent, one step for loading,
    one per feature and one for writing.
    """

    project: gws.Project
    """Project of the request."""
    request: gws.ExportRequest
    """Export request."""
    result: gws.ExportResult
    """Export result, filled by the exporter."""
    progressTotal: int = 1
    """Total number of progress steps."""
    progressStep: int = 0
    """Current progress step."""
    progressPercent: int = 0
    """Last reported progress, in percent."""

    @classmethod
    def run(cls, root: gws.Root, job: gws.Job):
        """Run an export job.

        Called by the job manager. Reads the request from the file given in the job payload.

        Args:
            root: Root object.
            job: The job.
        """
        request = gws.u.unserialize_from_path(job.payload.get('requestPath'))
        w = cls(root, job.user, job, request)
        w.work()

    def __init__(self, root: gws.Root, user: gws.User, job: Optional[gws.Job], request: gws.ExportRequest):
        """Create a worker.

        Args:
            root: Root object.
            user: User the export runs for.
            job: The job, or ``None`` to run without a job.
            request: Export request.
        """
        super().__init__(root, user, job)
        self.request = request
        self.result = gws.ExportResult(
            path='',
            mimeType='',
            numFiles=0,
            numFeaturesTotal=0,
            numFeaturesExported=0,
            errors=[],
        )

    def work(self):
        """Run the export and store the result in the job, if there is one.

        Raises:
            ``gws.NotFoundError``: If the project or the exporter does not exist.
            ``gws.ForbiddenError``: If the user cannot read the project.
            ``gws.JobTerminated``: If the job was cancelled or removed.
        """
        self.project = cast(gws.Project, self.user.require(self.request.projectUid, gws.ext.object.project))

        exporter = self.root.app.exporterMgr.get_exporter(
            [self.project, self.root.app],
            self.request.exporterUid,
            self.user,
        )
        if not exporter:
            raise gws.NotFoundError('exporter not found')

        # load + features + write
        self.progressTotal = 1 + len(self.request.features or []) + 1
        self.update_job(numSteps=100)

        features = self.load_features()
        self.notify('load')

        ea = gws.ExportArgs(
            exporter=exporter,
            features=features,
            shape=self.request.shape,
            project=self.project,
            user=self.user,
            notify=self.notify,
        )

        exporter.run(ea, self.result)
        self.update_job(state=gws.JobState.complete, result=self.result)

    def load_features(self):
        """Load the requested features.

        Features of models with the ``client`` export strategy are built from
        the request props, others are read from their models by uid. Features
        of unknown models and features that are not found are skipped with a
        warning.

        Returns:
            The features, in the order of the request.
        """
        if not self.request.features:
            return []

        self.result.numFeaturesTotal = len(self.request.features)

        models: dict[str, gws.Model] = {}
        uid_map: dict[str, set[str]] = {}
        fs_map: dict[str, gws.Feature] = {}

        mc = gws.ModelContext(op=gws.ModelOperation.export, user=self.user, project=self.project)

        for fp in self.request.features:
            model_uid = fp.modelUid
            feature_uid = fp.uid

            model = models.get(model_uid)
            if not model:
                model = self.root.app.modelMgr.get_model(model_uid, self.user)
                if not model:
                    gws.log.warning(f'model not found: {model_uid}')
                    continue
                models[model_uid] = model

            guid = gws.u.join_uid(model_uid, feature_uid)
            es = model.exportStrategy or gws.FeatureExportStrategy.load
            if es == gws.FeatureExportStrategy.client:
                fs_map[guid] = model.feature_from_props(fp, mc)
            else:
                uid_map.setdefault(model_uid, set()).add(feature_uid)

        for model_uid, feature_uids in uid_map.items():
            model = models[model_uid]
            fs = model.get_features(feature_uids, mc)
            for f in fs:
                guid = gws.u.join_uid(model_uid, f.uid())
                fs_map[guid] = f

        fs = []

        for fp in self.request.features:
            guid = gws.u.join_uid(fp.modelUid, fp.uid)
            f = fs_map.get(guid)
            if not f:
                gws.log.warning(f'feature not found: {guid}')
                continue
            fs.append(f)

        return fs

    def notify(self, event, details=None):
        """Advance the progress by one step and update the job.

        Args:
            event: Event name, not used.
            details: Event details, not used.

        Raises:
            ``gws.JobTerminated``: If the job was cancelled or removed.
        """
        job = self.get_job()
        if not job:
            return
        self.progressStep += 1
        p = int(self.progressStep / self.progressTotal * 100)
        if p > self.progressPercent:
            self.progressPercent = p
            self.update_job(numSteps=100, step=self.progressPercent)

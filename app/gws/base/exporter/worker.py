from typing import Optional, cast

import gws
import gws.base.model
import gws.base.job
import gws.lib.crs
import gws.gis.render
import gws.lib.image
import gws.lib.intl
import gws.lib.mime
import gws.lib.osx
import gws.lib.style
import gws.lib.uom


class Object(gws.base.job.worker.Object):
    project: gws.Project
    request: gws.ExportRequest
    tmpDir: str
    er: Optional[gws.ExportResult]
    features: list[gws.Feature]
    numFeaturesTotal: int

    @classmethod
    def run(cls, root: gws.Root, job: gws.Job):
        request = gws.u.unserialize_from_path(job.payload.get('requestPath'))
        w = cls(root, job.user, job, request)
        w.work()

    def __init__(self, root: gws.Root, user: gws.User, job: Optional[gws.Job], request: gws.ExportRequest):
        super().__init__(root, user, job)
        self.request = request
        self.er = None
        self.features = []
        self.numFeaturesTotal = 0

    def work(self):
        self.project = cast(gws.Project, self.user.require(self.request.projectUid, gws.ext.object.project))
        self.features = self.load_features()

        exporter = self.root.app.exporterMgr.get_exporter([self.project, self.root.app], self.request.exporterUid, self.user)
        if not exporter:
            raise gws.NotFoundError('exporter not found')

        ea = gws.ExportArgs(
            features=self.features,
            shape=self.request.shape,
            project=self.project,
            user=self.user,
            notify=self.notify,
        )

        self.er = exporter.run(ea)
        self.er.numFeaturesTotal = self.numFeaturesTotal
        self.update_job(state=gws.JobState.complete, payload=dict(result=self.er))

    def load_features(self):
        
        if not self.request.features:
            return []

        self.numFeaturesTotal = len(self.request.features)

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
        gws.log.debug(f'notify: {event} {details}')

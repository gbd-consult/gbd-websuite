import gws
import gws.lib.osx
from . import worker


class Object(gws.ExporterManager):
    """Exporter manager object."""

    def list_exporters(self, where, user):
        exporters = []
        titles = set()

        for obj in where:
            if not obj:
                continue
            for e in getattr(obj, 'exporters', []):
                if not user.can_use(e):
                    continue
                if e.title in titles:
                    continue
                titles.add(e.title)
                exporters.append(e)

        return sorted(exporters, key=lambda e: e.title)

    def get_exporter(self, where, uid, user):
        for e in self.list_exporters(where, user):
            if e.uid == uid:
                return e

    def start_export_job(self, request, user):
        mgr = self.root.app.jobMgr
        job = mgr.create_job(
            worker.Object,
            user,
            payload=dict(
                requestPath=gws.u.serialize_to_path(request, gws.u.ephemeral_path('export.pickle')),
            ),
        )
        job = mgr.schedule_job(job)
        return mgr.job_status_response(job)

    def exec_export(self, request, out_path):
        w = worker.Object(self.root, self.root.app.authMgr.systemUser, job=None, request=request)
        w.work()
        if w.result and w.result.path:
            gws.lib.osx.copy(w.result.path, out_path)
            return
        raise gws.Error('export failed')

    def handle_request(self, req, p):
        project = req.user.require_project(p.projectUid)
        exporter = self.get_exporter([project, self.root.app], p.exporterUid, req.user)
        if not exporter:
            raise gws.NotFoundError('exporter not found')
        return self.start_export_job(p, req.user)

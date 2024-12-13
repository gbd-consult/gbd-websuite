"""Printer manager."""

import gws
import gws.server.spool

from . import worker


class Object(gws.PrinterManager):
    def printers_for_project(self, project, user):
        ls = [p for p in project.printers if user.can_use(p)]
        ls.extend(p for p in self.root.app.printers if user.can_use(p))
        if ls:
            return ls
        return [self.root.app.defaultPrinter]

    def start_job(self, request, user):
        request_path = gws.u.serialize_to_path(request, gws.u.ephemeral_path('print.pickle'))

        job = self.root.app.jobMgr.create_job(
            user,
            worker.__file__ + '.main',
            gws.Data(
                requestPath=request_path,
                projectUid=request.projectUid
            )
        )

        if gws.server.spool.is_active():
            gws.server.spool.add(job)
            return job

        return self.root.app.jobMgr.run_job(job)

    def exec_print(self, request, user):
        w = worker.Object(self.root, '', request, user)
        return w.run()

    def cancel_job(self, job):
        return self.root.app.jobMgr.cancel_job(job)

    _url_path_suffix = '/gws.pdf'

    def job_response(self, job):
        res = self.root.app.jobMgr.job_response(job)
        res.resultUrl = gws.u.action_url_path('printerResult', jobUid=job.uid, projectUid=job.payload.get('projectUid')) + self._url_path_suffix
        return res

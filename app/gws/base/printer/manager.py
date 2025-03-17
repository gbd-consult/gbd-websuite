"""Printer manager."""

import gws
import gws.lib.mime
import gws.lib.osx

from . import worker


class Object(gws.PrinterManager):

    def start_print_job(self, request, user):
        mgr = self.root.app.jobMgr
        job = mgr.create_job(
            user,
            worker.Object,
            payload=dict(
                requestPath=gws.u.serialize_to_path(
                    request,
                    gws.u.ephemeral_path('print.pickle')
                ),
                outputPath='',
            )
        )
        job = mgr.schedule_job(job)
        return mgr.job_status_response(job)

    def exec_print(self, request, out_path):
        w = worker.Object(self.root, self.root.app.authMgr.systemUser, job=None, request=request)
        w.work()
        gws.lib.osx.copy(w.contentPath, out_path)

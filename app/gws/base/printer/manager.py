"""Printer manager."""

import gws
import gws.lib.mime

from . import worker


class Object(gws.PrinterManager):

    def start_print_job(self, request, user):
        mgr = self.root.app.jobMgr
        job = mgr.create_job(
            user,
            __file__ + '._worker',
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

    def exec_print(self, user, p):
        w = worker.Object(self.root, user, None, p)
        return w.run()


def _worker(root: gws.Root, job: gws.Job):
    request = gws.u.unserialize_from_path(job.payload.get('requestPath'))
    w = worker.Object(root, job.user, job, request)
    w.run()

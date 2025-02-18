"""Printer manager."""

import gws
import gws.lib.mime

from . import worker


class Object(gws.PrinterManager):

    def start_print_job(self, request, user):
        mgr = self.root.app.jobMgr
        job = mgr.create_job(user, worker.__file__ + '.main')
        job.requestPath = gws.u.serialize_to_path(request, gws.u.ephemeral_path('print.pickle'))
        job.outputMime = gws.lib.mime.PDF
        job = mgr.save_job(job)
        job = mgr.schedule_job(job)
        return mgr.status_response(job)

    def exec_print(self, request, user):
        w = worker.Object(self.root, '', request, user)
        return w.run()

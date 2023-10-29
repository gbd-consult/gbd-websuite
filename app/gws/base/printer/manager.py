"""Printer manager."""

import gws
import gws.lib.job
import gws.server.spool

import gws.types as t

from . import core, worker


class Object(gws.Node, gws.IPrinterManager):
    def printers_for_project(self, project, user):
        ls = [p for p in project.printers if user.can_use(p)]
        ls.extend(p for p in self.root.app.printers if user.can_use(p))
        if ls:
            return ls
        return [self.root.app.defaultPrinter]

    def start_job(self, request: core.Request, user: gws.IUser) -> gws.lib.job.Object:
        request_path = gws.serialize_to_path(request, gws.printtemp('print.pickle'))

        job = gws.lib.job.create(
            self.root,
            user=user,
            payload=dict(
                requestPath=request_path,
                projectUid=request.projectUid
            ),
            worker=worker.__name__ + '.worker')

        if gws.server.spool.is_active():
            gws.server.spool.add(job)
        else:
            gws.lib.job.run(self.root, job.uid)
        return gws.lib.job.get(self.root, job.uid)

    def get_job(self, uid: str, user: gws.IUser) -> t.Optional[gws.lib.job.Object]:
        job = gws.lib.job.get(self.root, uid)
        if not job:
            gws.log.error(f'JOB {uid}: not found')
            return
        if job.user.uid != user.uid:
            gws.log.error(f'JOB {uid}: wrong user (job={job.user.uid!r} user={user.uid!r})')
            return
        return job

    def run_job(self, request: core.Request, user: gws.IUser):
        w = worker.Object(self.root, '', request, user)
        return w.run()

    def cancel_job(self, job: gws.lib.job.Object):
        job.cancel()

    def result_path(self, job: gws.lib.job.Object) -> str:
        return job.payload.get('resultPath')

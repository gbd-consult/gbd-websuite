"""Printer manager."""

import gws
import gws.lib.job
import gws.server.spool


from . import core, worker


class Object(gws.PrinterManager):
    def printers_for_project(self, project, user):
        ls = [p for p in project.printers if user.can_use(p)]
        ls.extend(p for p in self.root.app.printers if user.can_use(p))
        if ls:
            return ls
        return [self.root.app.defaultPrinter]

    def start_job(self, request, user):
        request_path = gws.u.serialize_to_path(request, gws.u.printtemp('print.pickle'))

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

    def get_job(self, uid, user):
        job = gws.lib.job.get(self.root, uid)
        if not job:
            gws.log.error(f'JOB {uid}: not found')
            return
        if job.user.uid != user.uid:
            gws.log.error(f'JOB {uid}: wrong user (job={job.user.uid!r} user={user.uid!r})')
            return
        return job

    def run_job(self, request, user):
        w = worker.Object(self.root, '', request, user)
        return w.run()

    def cancel_job(self, job):
        job.cancel()

    def result_path(self, job):
        return job.payload.get('resultPath')

    def status(self, job) -> gws.PrintJobResponse:
        payload = job.payload

        def _progress():
            if job.state == gws.JobState.complete:
                return 100
            if job.state != gws.JobState.running:
                return 0
            num_steps = payload.get('numSteps', 0)
            if not num_steps:
                return 0
            step = payload.get('step', 0)
            return int(min(100.0, step * 100.0 / num_steps))

        _url_path_suffix = '/gws.pdf'

        return gws.PrintJobResponse(
            jobUid=job.uid,
            state=job.state,
            progress=_progress(),
            stepType=payload.get('stepType', ''),
            stepName=payload.get('stepName', ''),
            url=gws.u.action_url_path('printerResult', jobUid=job.uid, projectUid=payload.get('projectUid')) + _url_path_suffix
        )

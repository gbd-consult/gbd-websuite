"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.style

gws.ext.new.action('job')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass

class Object(gws.base.action.Object):

    @gws.ext.command.api('jobStatus')
    def job_status(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobResponse:
        mgr = self.root.app.jobMgr
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'JOB {p.jobUid}: not found')
        return mgr.status_response(job)

    @gws.ext.command.api('jobCancel')
    def job_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobResponse:
        mgr = self.root.app.jobMgr
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'JOB {p.jobUid}: not found')
        job = mgr.cancel_job(job)
        return mgr.status_response(job)

    @gws.ext.command.get('jobOutput')
    def job_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        mgr = self.root.app.jobMgr
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'JOB {p.jobUid}: not found')
        if job.state != gws.JobState.complete:
            raise gws.NotFoundError(f'JOB {p.jobUid}: wrong state {job.state!r}')
        if not job.outputPath:
            raise gws.NotFoundError(f'JOB {p.jobUid}: no output path')
        return gws.ContentResponse(contentPath=job.outputPath, mime=job.outputMime)

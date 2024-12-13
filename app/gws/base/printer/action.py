"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.style

from . import manager

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class JobRequest(gws.Request):
    jobUid: str


class CliParams(gws.CliParams):
    project: Optional[str]
    """project uid"""
    request: str
    """path to request.json"""
    output: str
    """output path"""


class Object(gws.base.action.Object):

    @gws.ext.command.api('printerStart')
    def printer_start(self, req: gws.WebRequester, p: gws.PrintRequest) -> gws.JobResponse:
        """Start a background print job"""

        mgr = self.root.app.printerMgr
        job = mgr.start_job(p, req.user)
        return mgr.job_response(job)

    @gws.ext.command.api('printerJobInfo')
    def printer_job_info(self, req: gws.WebRequester, p: JobRequest) -> gws.JobResponse:
        """Query the print job status"""

        mgr = self.root.app.printerMgr
        job = self._get_job(req, p)
        return mgr.job_response(job)

    @gws.ext.command.api('printerCancelJob')
    def printer_cancel_job(self, req: gws.WebRequester, p: JobRequest) -> gws.JobResponse:
        """Cancel a print job"""

        mgr = self.root.app.printerMgr
        job = self._get_job(req, p)
        job = mgr.cancel_job(job)
        return mgr.job_response(job)


    @gws.ext.command.get('printerResult')
    def printer_result(self, req: gws.WebRequester, p: JobRequest) -> gws.ContentResponse:
        """Get the result of a print job as bytes."""

        job = self._get_job(req, p)
        if job.state != gws.JobState.complete:
            raise gws.NotFoundError(f'printerResult {p.jobUid=} wrong {job.state=}')

        cp = job.payload.get('contentPath')
        if not cp:
            raise gws.NotFoundError(f'printerResult {p.jobUid=} no res_path')

        return gws.ContentResponse(contentPath=cp)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            p.request,
        )

        mgr = root.app.printerMgr
        res_path = mgr.exec_print(request, root.app.authMgr.systemUser)
        gws.u.write_file_b(p.output, gws.u.read_file_b(res_path))


    ##

    def _get_job(self, req: gws.WebRequester, p: JobRequest):
        job = self.root.app.jobMgr.get_job_for(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'{p.jobUid=} not found')
        return job

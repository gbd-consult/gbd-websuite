"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.job
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
    def start_print(self, req: gws.WebRequester, p: gws.PrintRequest) -> gws.PrintJobResponse:
        """Start a background print job"""

        job = self.root.app.printerMgr.start_job(p, req.user)
        return self.root.app.printerMgr.status(job)

    @gws.ext.command.api('printerStatus')
    def get_status(self, req: gws.WebRequester, p: JobRequest) -> gws.PrintJobResponse:
        """Query the print job status"""

        job = self.root.app.printerMgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'{p.jobUid=} not found')
        return self.root.app.printerMgr.status(job)

    @gws.ext.command.api('printerCancel')
    def cancel(self, req: gws.WebRequester, p: JobRequest) -> gws.PrintJobResponse:
        """Cancel a print job"""

        job = self.root.app.printerMgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'{p.jobUid=} not found')
        self.root.app.printerMgr.cancel_job(job)
        return self.root.app.printerMgr.status(job)

    @gws.ext.command.get('printerResult')
    def get_result(self, req: gws.WebRequester, p: JobRequest) -> gws.ContentResponse:
        """Get the result of a print job as a byte stream"""

        job = self.root.app.printerMgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.NotFoundError(f'printerResult {p.jobUid=} not found')
        if job.state != gws.JobState.complete:
            raise gws.NotFoundError(f'printerResult {p.jobUid=} wrong {job.state=}')

        res_path = self.root.app.printerMgr.result_path(job)
        if not res_path:
            raise gws.NotFoundError(f'printerResult {p.jobUid=} no res_path')

        return gws.ContentResponse(contentPath=res_path)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            p.request,
        )

        res_path = root.app.printerMgr.run_job(request, root.app.authMgr.systemUser)
        res = gws.u.read_file_b(res_path)
        gws.u.write_file_b(p.output, res)

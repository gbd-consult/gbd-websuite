"""Provides the printing API."""

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.job
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.style

import gws.types as t

from . import core, manager

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class JobRequest(gws.Request):
    jobUid: str


class CliParams(gws.CliParams):
    project: t.Optional[str]
    """project uid"""
    request: str
    """path to request.json"""
    output: str
    """output path"""


class Object(gws.base.action.Object):

    @gws.ext.command.api('printerStart')
    def start_print(self, req: gws.IWebRequester, p: core.Request) -> core.StatusResponse:
        """Start a background print job"""

        mgr = t.cast(manager.Object, self.root.app.printerMgr)
        job = mgr.start_job(p, req.user)
        return self.status(job)

    @gws.ext.command.api('printerStatus')
    def get_status(self, req: gws.IWebRequester, p: JobRequest) -> core.StatusResponse:
        """Query the print job status"""

        mgr = t.cast(manager.Object, self.root.app.printerMgr)
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.base.web.error.NotFound()
        return self.status(job)

    @gws.ext.command.api('printerCancel')
    def cancel(self, req: gws.IWebRequester, p: JobRequest) -> core.StatusResponse:
        """Cancel a print job"""

        mgr = t.cast(manager.Object, self.root.app.printerMgr)
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.base.web.error.NotFound()
        mgr.cancel_job(job)
        return self.status(job)

    @gws.ext.command.get('printerResult')
    def get_result(self, req: gws.IWebRequester, p: JobRequest) -> gws.ContentResponse:
        """Get the result of a print job as a byte stream"""

        mgr = t.cast(manager.Object, self.root.app.printerMgr)
        job = mgr.get_job(p.jobUid, req.user)
        if not job:
            raise gws.base.web.error.NotFound()
        if job.state != gws.JobState.complete:
            raise gws.base.web.error.NotFound()

        res_path = mgr.result_path(job)
        if not res_path:
            raise gws.base.web.error.NotFound()

        return gws.ContentResponse(path=res_path)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.base.printer.core.Request',
            p.request,
        )

        mgr = t.cast(manager.Object, self.root.app.printerMgr)
        res_path = mgr.run_job(request, root.app.authMgr.systemUser)
        res = gws.read_file_b(res_path)
        gws.write_file_b(p.output, res)

    def status(self, job: gws.lib.job.Object) -> core.StatusResponse:
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

        return core.StatusResponse(
            jobUid=job.uid,
            state=job.state,
            progress=_progress(),
            stepType=payload.get('stepType', ''),
            stepName=payload.get('stepName', ''),
            url=gws.action_url_path('printerResult', jobUid=job.uid, projectUid=payload.get('projectUid')) + _url_path_suffix
        )

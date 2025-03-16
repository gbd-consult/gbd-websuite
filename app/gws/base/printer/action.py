"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.base.web
import gws.config
import gws.lib.jsonx
import gws.lib.mime
import gws.lib.style

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class CliParams(gws.CliParams):
    project: Optional[str]
    """project uid"""
    request: str
    """path to request.json"""
    output: str
    """output path"""


class Object(gws.base.action.Object):

    @gws.ext.command.api('printerStart')
    def printer_start(self, req: gws.WebRequester, p: gws.PrintRequest) -> gws.JobStatusResponse:
        """Start a background print job"""
        return self.root.app.printerMgr.start_print_job(p, req.user)

    @gws.ext.command.api('printerStatus')
    def printer_status(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        res = self.root.app.jobMgr.handle_status_request(req, p)
        if res.status == gws.JobState.complete:
            res.output = {
                'url': gws.u.action_url_path('printerOutput', jobUid=res.jobUid) + '/gws.pdf'
            }
        return res

    @gws.ext.command.api('printerCancel')
    def printer_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        return self.root.app.jobMgr.handle_cancel_request(req, p)

    @gws.ext.command.get('printerOutput')
    def printer_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        job = self.root.app.jobMgr.require_job(req, p)

        if job.state != gws.JobState.complete:
            raise gws.NotFoundError(f'JOB {p.jobUid}: wrong state {job.state!r}')

        out = job.payload.get('outputPath')
        if not out:
            raise gws.NotFoundError(f'JOB {p.jobUid}: no output path')
        return gws.ContentResponse(contentPath=out, mime=gws.lib.mime.PDF)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            path=p.request,
        )

        res_path = root.app.printerMgr.exec_print(request, root.app.authMgr.systemUser)
        gws.u.write_file_b(p.output, gws.u.read_file_b(res_path))

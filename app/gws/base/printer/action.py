"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.mime

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    """Configuration for the printer action."""

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
        if res.state == gws.JobState.complete:
            pr = gws.PrintResult(self.root.app.jobMgr.require_result(req, p))
            ext = gws.lib.mime.extension_for(pr.mime) or 'bin'
            res.output = {
                'url': gws.u.action_url_path('printerOutput', jobUid=res.jobUid) + f'/gws.{ext}',
            }
        return res

    @gws.ext.command.api('printerCancel')
    def printer_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        return self.root.app.jobMgr.handle_cancel_request(req, p)

    @gws.ext.command.get('printerOutput')
    def printer_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        pr = gws.PrintResult(self.root.app.jobMgr.require_result(req, p))
        return gws.ContentResponse(contentPath=pr.path, mime=pr.mime)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            path=p.request,
        )

        root.app.printerMgr.exec_print(cast(gws.PrintRequest, request), p.output)

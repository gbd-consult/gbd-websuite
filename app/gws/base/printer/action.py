"""Printer action."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.mime


@gws.ext.config.action('printer')
class Config(gws.base.action.Config):
    """Runs print jobs in the background and returns their output."""

    pass


@gws.ext.props.action('printer')
class Props(gws.base.action.Props):
    pass


class CliParams(gws.CliParams):
    """Parameters for the ``printerPrint`` command."""

    project: Optional[str]
    """Project uid."""
    request: str
    """Path to a JSON file with the print request."""
    output: str
    """Output path."""


@gws.ext.object.action('printer')
class Object(gws.base.action.Object):
    """Printer action."""

    @gws.ext.command.api('printerStart')
    def printer_start(self, req: gws.WebRequester, p: gws.PrintRequest) -> gws.JobStatusResponse:
        """Start a background print job."""
        return self.root.app.printerMgr.start_print_job(p, req.user)

    @gws.ext.command.api('printerStatus')
    def printer_status(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        """Return the status of a print job."""

        res = self.root.app.jobMgr.handle_status_request(req, p)
        if res.state == gws.JobState.complete:
            pr = gws.PrintResult(self.root.app.jobMgr.require_result(req, p))
            ext = gws.lib.mime.extension_for(pr.mimeType) or 'bin'
            res.output = {
                'url': gws.u.action_url_path('printerOutput', jobUid=res.jobUid) + f'/gws.{ext}',
            }
        return res

    @gws.ext.command.api('printerCancel')
    def printer_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        """Cancel a print job."""

        return self.root.app.jobMgr.handle_cancel_request(req, p)

    @gws.ext.command.get('printerOutput')
    def printer_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        """Return the result file of a completed print job."""

        pr = gws.PrintResult(self.root.app.jobMgr.require_result(req, p))
        return gws.ContentResponse(contentPath=pr.path, mimeType=pr.mimeType)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliParams):
        """Print a request from a JSON file."""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.PrintRequest',
            path=p.request,
        )

        root.app.printerMgr.exec_print(cast(gws.PrintRequest, request), p.output)

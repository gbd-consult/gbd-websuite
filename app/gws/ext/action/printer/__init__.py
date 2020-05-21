"""Provides the printing API."""

import gws.common.action
import gws.common.printer.job
import gws.tools.job
import gws.common.printer.types as pt
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Print action"""
    pass


class Object(gws.common.action.Object):

    def api_print(self, req: t.IRequest, p: pt.PrintParams) -> gws.tools.job.StatusResponse:
        """Start a backround print job"""

        return gws.common.printer.job.start(req, p)

    def api_snapshot(self, req: t.IRequest, p: pt.PrintParams) -> gws.tools.job.StatusResponse:
        """Start a backround snapshot job"""

        return gws.common.printer.job.start(req, p)

    def api_status(self, req: t.IRequest, p: gws.tools.job.StatusParams) -> gws.tools.job.StatusResponse:
        """Query the print job status"""

        r = gws.tools.job.status_request(req, p)
        if not r:
            raise gws.web.error.NotFound()
        return r

    def api_cancel(self, req: t.IRequest, p: gws.tools.job.StatusParams) -> gws.tools.job.StatusResponse:
        """Cancel a print job"""

        r = gws.tools.job.cancel_request(req, p)
        if not r:
            raise gws.web.error.NotFound()
        return r

    def http_get_result(self, req: t.IRequest, p: gws.tools.job.StatusParams) -> t.Response:
        job = gws.tools.job.get_for(req.user, p.jobUid)
        if not job or job.state != gws.tools.job.State.complete:
            raise gws.web.error.NotFound()
        return t.FileResponse(mime='application/pdf', path=job.result)

"""Provides the printing API."""

import gws.base.action
import gws.base.printer.job as pj
import gws.lib.job
import gws.lib.mime
import gws.base.printer.types as pt
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Print action"""
    pass


class Object(gws.base.action.Object):

    def api_print(self, req: t.IRequest, p: pt.PrintParams) -> pj.StatusResponse:
        """Start a backround print job"""

        job = pj.start(req, p)
        return pj.status(job)

    def api_snapshot(self, req: t.IRequest, p: pt.PrintParams) -> pj.StatusResponse:
        """Start a backround snapshot job"""

        job = pj.start(req, p)
        return pj.status(job)

    def api_status(self, req: t.IRequest, p: pj.StatusParams) -> pj.StatusResponse:
        """Query the print job status"""

        job = gws.lib.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.web.error.NotFound()

        return pj.status(job)

    def api_cancel(self, req: t.IRequest, p: pj.StatusParams) -> pj.StatusResponse:
        """Cancel a print job"""

        job = gws.lib.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.web.error.NotFound()

        job.cancel()

        return pj.status(job)

    def http_get_result(self, req: t.IRequest, p: pj.StatusParams) -> t.Response:
        job = gws.lib.job.get_for(req.user, p.jobUid)
        if not job or job.state != gws.lib.job.State.complete:
            raise gws.web.error.NotFound()

        path = job.result['path']
        return t.FileResponse(mime=gws.lib.mime.for_path(path), path=path)


"""Provides the printing API."""

import gws
import gws.base.api
import gws.base.web.error
import gws.lib.job
import gws.lib.mime

from . import job, types


@gws.ext.Object('action.printer')
class Object(gws.base.api.action.Object):

    @gws.ext.command('api.printer.startPrint')
    def start_print(self, req: gws.IWebRequest, p: types.Params) -> types.StatusResponse:
        """Start a backround print job"""

        j = job.start(req, p)
        return job.status(j)

    @gws.ext.command('api.printer.startSnapshot')
    def start_snapshot(self, req: gws.IWebRequest, p: types.Params) -> types.StatusResponse:
        """Start a backround snapshot job"""

        j = job.start(req, p)
        return job.status(j)

    @gws.ext.command('api.printer.getStatus')
    def get_status(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Query the print job status"""

        j = gws.lib.job.get_for(req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        return job.status(j)

    @gws.ext.command('api.printer.cancelPrinting')
    def cancel(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Cancel a print job"""

        j = gws.lib.job.get_for(req.user, p.jobUid)
        if not j:
            raise gws.base.web.error.NotFound()

        j.cancel()

        return job.status(j)

    @gws.ext.command('get.printer.getResult')
    def get_result(self, req: gws.IWebRequest, p: types.StatusParams) -> gws.ContentResponse:
        j = gws.lib.job.get_for(req.user, p.jobUid)
        if not j or j.state != gws.lib.job.State.complete:
            raise gws.base.web.error.NotFound()

        path = j.result['path']
        return gws.ContentResponse(mime=gws.lib.mime.for_path(path), path=path)

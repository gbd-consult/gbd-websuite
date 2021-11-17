"""Provides the printing API."""

import gws
import gws.base.api
import gws.base.web.error
import gws.config
import gws.lib.job
import gws.lib.json2
import gws.lib.mime
import gws.types as t

from . import job, types


class CliPrintParams(gws.CliParams):
    project: t.Optional[str]  #: project uid
    params: str  #: path to params.json
    output: str  #: output path


@gws.ext.Object('action.printer')
class Object(gws.base.api.action.Object):

    @gws.ext.command('api.printer.startPrint')
    def start_print(self, req: gws.IWebRequest, p: types.Params) -> types.StatusResponse:
        """Start a backround print job"""

        j = job.start(self.root, req, p)
        return job.status(j)

    @gws.ext.command('api.printer.getStatus')
    def get_status(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Query the print job status"""

        j = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        return job.status(j)

    @gws.ext.command('api.printer.cancelPrinting')
    def cancel(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Cancel a print job"""

        j = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not j:
            raise gws.base.web.error.NotFound()

        j.cancel()

        return job.status(j)

    @gws.ext.command('get.printer.getResult')
    def get_result(self, req: gws.IWebRequest, p: types.StatusParams) -> gws.ContentResponse:
        j = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not j or j.state != gws.lib.job.State.complete:
            raise gws.base.web.error.NotFound()

        path = j.result['path']
        return gws.ContentResponse(mime=gws.lib.mime.for_path(path), path=path)

    @gws.ext.command('cli.printer.print')
    def print(self, p: CliPrintParams):
        root = gws.config.load()
        params = root.specs.read_value(
            gws.lib.json2.from_path(p.params),
            'gws.base.printer.types.Params',
            p.params
        )
        res_path = job.run(root, params, p.project, user=root.application.auth.system_user)
        res = gws.read_file_b(res_path)
        gws.write_file_b(p.output, res)

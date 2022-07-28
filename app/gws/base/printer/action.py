"""Provides the printing API."""

import gws
import gws.base.api
import gws.base.web.error
import gws.config
import gws.lib.job
import gws.lib.json2
import gws.lib.mime
import gws.types as t

from . import core, types


class CliPrintParams(gws.CliParams):
    project: t.Optional[str]  #: project uid
    params: str  #: path to params.json
    output: str  #: output path


@gws.ext.object.action('printer')
class Object(gws.base.api.action.Object):

    @gws.ext.command.api('printerStart')
    def start_print(self, req: gws.IWebRequest, p: types.Params) -> types.StatusResponse:
        """Start a backround print job"""

        job = core.start_job(self.root, req, p)
        return core.job_status(job)

    @gws.ext.command.api('printerStatus')
    def get_status(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Query the print job status"""

        job = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        return core.job_status(job)

    @gws.ext.command.api('printerCancel')
    def cancel(self, req: gws.IWebRequest, p: types.StatusParams) -> types.StatusResponse:
        """Cancel a print job"""

        job = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not job:
            raise gws.base.web.error.NotFound()

        job.cancel()
        return core.job_status(job)

    @gws.ext.command.get('printerResult')
    def get_result(self, req: gws.IWebRequest, p: types.StatusParams) -> gws.ContentResponse:
        """Get the result of a print job as a byte stream"""

        job = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not job or job.state != gws.lib.job.State.complete:
            raise gws.base.web.error.NotFound()

        res_path = core.job_result_path(job)
        if not res_path:
            raise gws.base.web.error.NotFound()

        return gws.ContentResponse(path=res_path)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliPrintParams):
        """Print using the specified params"""

        root = gws.config.load()
        params = root.specs.read_value(
            gws.lib.json2.from_path(p.params),
            'gws.base.printer.types.Params',
            p.params
        )
        res_path = core.run_job(root, params, p.project, user=root.application.auth.system_user)
        res = gws.read_file_b(res_path)
        gws.write_file_b(p.output, res)

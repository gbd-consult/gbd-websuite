"""Provides the printing API."""

import gws
import gws.base.action
import gws.base.web.error
import gws.config
import gws.lib.job
import gws.lib.jsonx
import gws.lib.mime
import gws.types as t

from . import core, job

gws.ext.new.action('printer')


class Config(gws.base.action.Config):
    pass


class Props(gws.base.action.Props):
    pass


class CliPrintParams(gws.CliParams):
    project: t.Optional[str]
    """project uid"""
    params: str
    """path to params.json"""
    output: str
    """output path"""


class Object(gws.base.action.Object):

    @gws.ext.command.api('printerStart')
    def start_print(self, req: gws.IWebRequester, p: core.Params) -> core.StatusResponse:
        """Start a backround print job"""

        jb = job.start_job(self.root, req, p)
        return job.job_status(jb)

    @gws.ext.command.api('printerStatus')
    def get_status(self, req: gws.IWebRequester, p: core.StatusParams) -> core.StatusResponse:
        """Query the print job status"""

        jb = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not jb:
            raise gws.base.web.error.NotFound()

        return job.job_status(jb)

    @gws.ext.command.api('printerCancel')
    def cancel(self, req: gws.IWebRequester, p: core.StatusParams) -> core.StatusResponse:
        """Cancel a print job"""

        jb = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not jb:
            raise gws.base.web.error.NotFound()

        jb.cancel()
        return job.job_status(jb)

    @gws.ext.command.get('printerResult')
    def get_result(self, req: gws.IWebRequester, p: core.StatusParams) -> gws.ContentResponse:
        """Get the result of a print job as a byte stream"""

        jb = gws.lib.job.get_for(self.root, req.user, p.jobUid)
        if not jb or jb.state != gws.lib.job.State.complete:
            raise gws.base.web.error.NotFound()

        res_path = job.job_result_path(jb)
        if not res_path:
            raise gws.base.web.error.NotFound()

        return gws.ContentResponse(path=res_path)

    @gws.ext.command.cli('printerPrint')
    def print(self, p: CliPrintParams):
        """Print using the specified params"""

        root = gws.config.load()
        params = root.specs.read(
            gws.lib.jsonx.from_path(p.params),
            core.Params,
            p.params
        )
        res_path = job.run_job(root, params, p.project, user=root.app.auth.systemUser)
        res = gws.read_file_b(res_path)
        gws.write_file_b(p.output, res)

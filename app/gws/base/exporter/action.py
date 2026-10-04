"""Exporter action."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config
import gws.lib.jsonx
import gws.lib.mime


@gws.ext.config.action('exporter')
class Config(gws.base.action.Config):
    """Feature export in the client, run as background jobs."""

    pass


@gws.ext.props.action('exporter')
class Props(gws.base.action.Props):
    pass


class CliParams(gws.CliParams):
    """Parameters of the ``gws exporter export`` command."""

    project: Optional[str]
    """Project uid. Not used, the project is taken from the request."""
    request: str
    """Path to a JSON file with a ``gws.ExportRequest``."""
    output: str
    """Path to copy the export result to."""


@gws.ext.object.action('exporter')
class Object(gws.base.action.Object):
    """Exporter action."""

    @gws.ext.command.api('exporterStart')
    def exporter_start(self, req: gws.WebRequester, p: gws.ExportRequest) -> gws.JobStatusResponse:
        """Start a background export job."""
        return self.root.app.exporterMgr.start_export_job(p, req.user)

    @gws.ext.command.api('exporterStatus')
    def exporter_status(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        """Return the status of an export job."""
        res = self.root.app.jobMgr.handle_status_request(req, p)
        if res.state == gws.JobState.complete:
            er = gws.ExportResult(self.root.app.jobMgr.require_result(req, p))
            res.output = {
                'numFiles': er.numFiles,
                'numFeaturesTotal': er.numFeaturesTotal,
                'numFeaturesExported': er.numFeaturesExported,
            }
            if er.path:
                ext = gws.lib.mime.extension_for(er.mimeType) or 'bin'
                res.output['url'] = gws.u.action_url_path('exporterOutput', projectUid=p.projectUid, jobUid=res.jobUid) + f'/gws.{ext}'

        return res

    @gws.ext.command.api('exporterCancel')
    def exporter_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        """Cancel an export job."""
        return self.root.app.jobMgr.handle_cancel_request(req, p)

    @gws.ext.command.get('exporterOutput')
    def exporter_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        """Return the result file of a completed export job."""
        er = gws.ExportResult(self.root.app.jobMgr.require_result(req, p))
        if not er.path:
            raise gws.NotFoundError('export result not found')
        return gws.ContentResponse(
            contentPath=er.path,
            mimeType=er.mimeType,
        )

    @gws.ext.command.cli('exporterExport')
    def do_export(self, p: CliParams):
        """Run an export from the command line."""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.ExportRequest',
            path=p.request,
        )

        root.app.exporterMgr.exec_export(cast(gws.ExportRequest, request), p.output)

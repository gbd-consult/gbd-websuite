"""Provides the printing API."""

from typing import Optional, cast

import gws
import gws.base.action
import gws.config
import gws.lib.jsonx
import gws.lib.osx
import gws.lib.mime

gws.ext.new.action('exporter')


class Config(gws.base.action.Config):
    """Configuration for the exporter action. (added in 8.4)"""

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
    @gws.ext.command.api('exporterStart')
    def exporter_start(self, req: gws.WebRequester, p: gws.ExportRequest) -> gws.JobStatusResponse:
        """Start a background export job"""
        return self.root.app.exporterMgr.start_export_job(p, req.user)

    @gws.ext.command.api('exporterStatus')
    def exporter_status(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        res = self.root.app.jobMgr.handle_status_request(req, p)
        if res.state == gws.JobState.complete:
            er = self._result(req, p)
            res.output = {
                'numFiles': er.numFiles,
                'numFeaturesTotal': er.numFeaturesTotal,
                'numFeaturesExported': er.numFeaturesExported,
            }
            if er.path:
                ext = gws.lib.mime.extension_for(er.mime) or 'bin'
                res.output['url'] = gws.u.action_url_path('exporterOutput', jobUid=res.jobUid) + f'/gws.{ext}'

        return res

    @gws.ext.command.api('exporterCancel')
    def exporter_cancel(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.JobStatusResponse:
        return self.root.app.jobMgr.handle_cancel_request(req, p)

    @gws.ext.command.get('exporterOutput')
    def exporter_output(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ContentResponse:
        er = self._result(req, p)
        if not er.path:
            raise gws.NotFoundError('export result not found')
        return gws.ContentResponse(
            contentPath=er.path,
            mime=er.mime,
        )

    @gws.ext.command.cli('exporterExport')
    def do_export(self, p: CliParams):
        """Export using the specified params"""

        root = gws.config.load()
        request = root.specs.read(
            gws.lib.jsonx.from_path(p.request),
            'gws.ExportRequest',
            path=p.request,
        )

        root.app.exporterMgr.exec_export(cast(gws.ExportRequest, request), p.output)

    def _result(self, req: gws.WebRequester, p: gws.JobRequest) -> gws.ExportResult:
        job = self.root.app.jobMgr.require_job(req, p)

        if job.state != gws.JobState.complete:
            raise gws.NotFoundError(f'JOB {p.jobUid}: wrong state {job.state!r}')

        res = job.payload.get('result')
        if not res:
            raise gws.NotFoundError(f'JOB {p.jobUid}: no result')

        return gws.ExportResult(res)

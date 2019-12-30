import gws.web
import gws.common.printer.service as service
import gws.common.printer.types as pt
import gws.types as t


class Config(t.WithTypeAndAccess):
    """Print action"""
    pass


class Object(gws.ActionObject):

    def api_print(self, req: t.IRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround print job"""

        req.require_project(p.projectUid)
        tpl: t.ITemplate = req.require('gws.ext.template', p.templateUid)

        for sec in p.sections:
            sec.attributes = tpl.normalize_user_data(sec.get('attributes'))

        return service.start_job(req, p)

    def api_snapshot(self, req: t.IRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround snapshot job"""

        req.require_project(p.projectUid)
        return service.start_job(req, p)

    def api_query(self, req: t.IRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Query the print job status"""

        return service.query_job(req, p)

    def api_cancel(self, req: t.IRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Cancel a print job"""

        return service.cancel_job(req, p)

    def http_get_result(self, req: t.IRequest, p: pt.PrinterQueryParams) -> t.HttpResponse:
        return service.job_result(req, p)

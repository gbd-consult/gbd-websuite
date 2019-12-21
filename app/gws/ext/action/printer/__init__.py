import gws
import gws.web
import gws.common.printer.service as service
import gws.common.printer.types as pt
import gws.types as t


class Config(t.WithTypeAndAccess):
    """Print action"""
    pass


class Object(gws.ActionObject):

    def api_print(self, req: gws.web.AuthRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround print job"""

        req.require_project(p.projectUid)
        template = req.require('gws.ext.template', p.templateUid)

        for sec in p.sections:
            sec.data = _validate_data(sec.get('data'), template)

        return service.start_job(req, p)

    def api_snapshot(self, req: gws.web.AuthRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround snapshot job"""

        req.require_project(p.projectUid)
        return service.start_job(req, p)

    def api_query(self, req: gws.web.AuthRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Query the print job status"""

        return service.query_job(req, p)

    def api_cancel(self, req: gws.web.AuthRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Cancel a print job"""

        return service.cancel_job(req, p)

    def http_get_result(self, req: gws.web.AuthRequest, p: pt.PrinterQueryParams) -> t.HttpResponse:
        return service.job_result(req, p)


def _validate_data(data, template):
    if not data or not template.data_model:
        return {}

    d = {}
    for attr in template.data_model:
        if attr.name in data:
            # @TODO convert to type
            d[attr.name] = gws.as_str(data[attr.name])

    return d

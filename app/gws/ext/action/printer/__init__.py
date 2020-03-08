"""Provides the printing API."""

import gws.common.action
import gws.common.printer.control as control
import gws.common.printer.types as pt
import gws.web.error

import gws.types as t


class Config(t.WithTypeAndAccess):
    """Print action"""
    pass


class Object(gws.common.action.Object):

    def api_print(self, req: t.IRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround print job"""

        return control.start_job(req, p)

    def api_snapshot(self, req: t.IRequest, p: pt.PrintParams) -> pt.PrinterResponse:
        """Start a backround snapshot job"""

        return control.start_job(req, p)

    def api_query(self, req: t.IRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Query the print job status"""

        return control.query_job(req, p)

    def api_cancel(self, req: t.IRequest, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
        """Cancel a print job"""

        return control.cancel_job(req, p)

    def http_get_result(self, req: t.IRequest, p: pt.PrinterQueryParams) -> t.Response:
        return control.job_result(req, p)

import gws
import gws.gis.feature
import gws.server.spool
import gws.tools.job
import gws.common.printer.job
import gws.types as t
import gws.web
from . import types as pt


def start_job(req, p: pt.PrintParams) -> pt.PrinterResponse:
    job = gws.common.printer.job.create(req, p)
    gws.server.spool.add(job)
    return pt.PrinterResponse({
        'jobUid': job.uid,
        'state': job.state
    })


def query_job(req, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
    job = _get_job(req, p.jobUid)

    progress = 0
    if job.steps and job.state == gws.tools.job.State.running:
        progress = min(100, int(job.step * 100 / job.steps))

    return pt.PrinterResponse({
        'jobUid': job.uid,
        'state': job.state,
        'progress': progress,
        'otype': job.otype or '',
        'oname': job.oname or '',
        'url': gws.SERVER_ENDPOINT + '?cmd=assetHttpGetResult&jobUid=' + job.uid,
    })


def cancel_job(req, p: pt.PrinterQueryParams) -> pt.PrinterResponse:
    job = _get_job(req, p.jobUid)
    job.cancel()
    return pt.PrinterResponse({
        'jobUid': job.uid,
        'state': gws.tools.job.State.cancel,
    })


def job_result(req, p: pt.PrinterQueryParams) -> t.HttpResponse:
    job = _get_job(req, p.jobUid)
    if job.state != gws.tools.job.State.complete:
        raise gws.web.error.NotFound()
    with open(job.result, 'rb') as fp:
        pdf = fp.read()
    return t.HttpResponse({
        'mimeType': 'application/pdf',
        'content': pdf
    })


def _get_job(req, uid):
    job = gws.tools.job.get_for(req.user, uid)
    if job:
        return job
    raise gws.web.error.NotFound()

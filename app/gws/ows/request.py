import gws.tools.net
import gws.tools.xml3

from . import error


def raw_get(url, **kwargs):
    # the reason to use lax is that we want an exception text from the server
    # even if the status != 200

    kwargs['lax'] = True

    resp = gws.tools.net.http_request(url, **kwargs)
    status = resp.status_code

    _check_ows_error(resp.text)

    if status != 200:
        resp.raise_for_status()

    return resp

def get(url, service, request, **kwargs):
    """Get a raw service response"""

    params = kwargs.get('params') or {}
    params['SERVICE'] = service.upper()
    params['REQUEST'] = request

    # some guys accept only uppercase params
    params = {k.upper(): v for k, v in params.items()}

    kwargs['params'] = params

    return raw_get(url, **kwargs)


def get_text(url, service, request, **kwargs):
    resp = get(url, service, request, **kwargs)
    return resp.text


_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def _check_ows_error(text):
    s = text[:2048].lower()
    for e in _ows_error_strings:
        if e.lower() in s:
            raise error.Error(text.strip())

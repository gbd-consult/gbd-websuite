import gws.lib.net
import gws.lib.xml2

from . import error

_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def raw_get(url, **kwargs):
    # the reason to use lax is that we want an exception text from the server
    # even if the status != 200

    kwargs['lax'] = True

    try:
        resp = gws.lib.net.http_request(url, **kwargs)
    except gws.lib.net.Error as e:
        raise error.Error('http error') from e

    status = resp.status_code

    # check for an ows error (no matter what status code says)
    # we can get big image responses here, so be careful and don't blindly decode them

    if resp.content.startswith(b'<') or 'xml' in resp.content_type:
        text = str(resp.content[:1024], encoding='utf8', errors='ignore').lower()
        for msg in _ows_error_strings:
            if msg.lower() in text:
                raise error.Error(resp.text[:1024])

    if status != 200:
        raise error.Error(f'HTTP error: {resp.status_code!r}')

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

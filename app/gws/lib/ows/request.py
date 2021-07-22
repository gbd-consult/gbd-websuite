import gws.lib.net
import gws.lib.xml2

from . import error

_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def raw_get(url, **kwargs):
    res: gws.lib.net.HTTPResponse = gws.lib.net.http_request(url, **kwargs)

    status = res.status_code

    # check for an ows error (no matter what status code says)
    # we can get big image responses here, so be careful and don't blindly decode them

    if res.content.startswith(b'<') or 'xml' in res.content_type:
        text = str(res.content[:1024], encoding='utf8', errors='ignore').lower()
        for msg in _ows_error_strings:
            if msg.lower() in text:
                raise error.Error(res.text[:1024])

    if status != 200:
        raise error.Error(f'HTTP error: {res.status_code!r}')

    return res


def get(url, service, request, **kwargs):
    """Get a raw service response"""

    params = kwargs.get('params') or {}
    params['SERVICE'] = service.upper()
    params['REQUEST'] = request

    # some guys accept only uppercase params
    params = {k.upper(): v for k, v in params.items()}

    kwargs['params'] = params

    return raw_get(url, **kwargs)


def get_text(url, service, request, **kwargs) -> str:
    res = get(url, service, request, **kwargs)
    return res.text


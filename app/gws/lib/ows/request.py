import gws.lib.net
import gws.lib.xml2

from . import error

_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def raw_get(url, **kwargs):
    res: gws.lib.net.HTTPResponse = gws.lib.net.http_request(url, **kwargs)

    status = res.status_code

    # check for an ows error (no matter what the status code says)
    # we can get big image responses here, so be careful and don't blindly decode everything

    if res.content.startswith(b'<') or 'xml' in res.content_type:
        text = str(res.content[:1024], encoding='utf8', errors='ignore').lower()
        for msg in _ows_error_strings:
            if msg.lower() in text:
                raise error.Error(res.text[:1024])

    if status != 200:
        raise error.Error(f'HTTP error: {res.status_code!r}')

    return res


def get(url, service, verb, **kwargs):
    """Get a raw service response"""

    params = kwargs.pop('params', None) or {}

    # some guys accept only uppercase params
    params = {k.upper(): v for k, v in params.items()}

    params.setdefault('SERVICE', service.upper())
    params.setdefault('REQUEST', verb)

    return raw_get(url, params=params, **kwargs)


def get_text(url, service, verb, **kwargs) -> str:
    res = get(url, service, verb, **kwargs)
    return res.text


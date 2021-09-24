import gws.lib.net
import gws.lib.xml2

from . import error

_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def get_url(url: str, **kwargs) -> gws.lib.net.HTTPResponse:
    res = gws.lib.net.http_request(url, **kwargs)

    # some folks serve OWS error documents with the status 200
    # therefore, we check for an ows error message, no matter what the status code says
    # we can get big image responses here, so be careful and don't blindly decode everything

    if res.content.startswith(b'<') or 'xml' in res.content_type:
        text = str(res.content[:1024], encoding='utf8', errors='ignore')
        text_lower = text.lower()
        for err_string in _ows_error_strings:
            if err_string.lower() in text_lower:
                raise error.Error(text)

    if not res.ok:
        raise error.Error(f'HTTP error: {res.status_code!r} {res.text!r}')

    return res


def get(url: str, protocol: gws.OwsProtocol, verb: gws.OwsVerb, **kwargs) -> gws.lib.net.HTTPResponse:
    """Get a raw service response"""

    params = kwargs.pop('params', None) or {}

    # some folks only accept uppercase params
    params = {k.upper(): v for k, v in params.items()}

    params.setdefault('SERVICE', str(protocol).upper())
    params.setdefault('REQUEST', verb)

    return get_url(url, params=params, **kwargs)


def get_text(url: str, protocol: gws.OwsProtocol, verb: gws.OwsVerb, **kwargs) -> str:
    res = get(url, protocol, verb, **kwargs)
    return res.text

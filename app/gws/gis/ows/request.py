import gws.lib.net

from . import error

_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


class Args(gws.Data):
    method: gws.RequestMethod
    params: dict
    protocol: gws.OwsProtocol
    url: str
    verb: gws.OwsVerb
    version: str


def _get_url(url: str, **kwargs) -> gws.lib.net.HTTPResponse:
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


def get(args: Args, **kwargs) -> gws.lib.net.HTTPResponse:
    """Get a raw service response"""

    params = {
        'SERVICE': str(args.protocol).upper(),
        'REQUEST': args.verb,
        'VERSION': args.version,
    }
    if args.params:
        params.update(args.params)

    return _get_url(args.url, method=args.method, params=params, **kwargs)


def get_text(args: Args, **kwargs) -> str:
    res = get(args, **kwargs)
    return res.text

"""HTTP requests to OWS services."""

import gws
import gws.lib.net


_ows_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


class Args(gws.Data):
    """Arguments for an OWS request."""

    method: gws.RequestMethod
    """Request method, ``GET`` by default."""
    headers: dict
    """HTTP headers."""
    params: dict
    """Additional request parameters."""
    protocol: gws.OwsProtocol
    """Service protocol, sent as ``SERVICE``."""
    url: str
    """Service URL."""
    verb: gws.OwsVerb
    """Request type, sent as ``REQUEST``."""
    version: str
    """Service version, sent as ``VERSION`` if not empty."""


def get_url(url: str, **kwargs) -> gws.lib.net.HTTPResponse:
    """Send an HTTP request to an OWS service.

    The response is checked for an OWS exception document, regardless of the
    HTTP status, since some servers return errors with status 200.

    Args:
        url: Request URL.
        **kwargs: Options for ``gws.lib.net.http_request``.

    Returns:
        The HTTP response.

    Raises:
        ``gws.ExternalServiceError``: If the service returns an exception document or the request fails.
    """

    res = gws.lib.net.http_request(url, **kwargs)

    # some folks serve OWS error documents with the status 200
    # therefore, we check for an ows error message, no matter what the status code says
    # we can get big image responses here, so be careful and don't blindly decode everything

    if res.content.startswith(b'<') or 'xml' in res.content_type:
        text = str(res.content[:1024], encoding='utf8', errors='ignore')
        text_lower = text.lower()
        for err_string in _ows_error_strings:
            if err_string.lower() in text_lower:
                raise gws.ExternalServiceError(text)
    try:
        res.raise_if_failed()
    except gws.lib.net.HTTPError as exc:
        raise gws.ExternalServiceError(f'network error: {url!r} -> {exc}') from exc
    return res


def get(args: Args, **kwargs) -> gws.lib.net.HTTPResponse:
    """Send an OWS request and return the raw response.

    Args:
        args: Request arguments.
        **kwargs: Options for ``gws.lib.net.http_request``.

    Returns:
        The HTTP response.

    Raises:
        ``gws.ExternalServiceError``: If the service returns an exception document or the request fails.
    """

    params = {
        'SERVICE': str(args.protocol).upper(),
        'REQUEST': args.verb,
    }
    if args.version:
        params['VERSION'] = args.version
    if args.params:
        params.update(gws.u.to_upper_dict(args.params))

    return get_url(args.url, method=args.method or gws.RequestMethod.GET, params=params, headers=args.headers, **kwargs)


def get_text(args: Args, **kwargs) -> str:
    """Send an OWS request and return the response text.

    Args:
        args: Request arguments.
        **kwargs: Options for ``gws.lib.net.http_request``.

    Returns:
        The response text.

    Raises:
        ``gws.ExternalServiceError``: If the service returns an exception document or the request fails.
    """

    res = get(args, **kwargs)
    return res.text

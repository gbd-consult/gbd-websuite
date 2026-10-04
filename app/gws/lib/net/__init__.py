"""URL utilities and an HTTP client.

URLs:

``parse_url`` splits a URL into a ``Url`` object (scheme, host, port, path, query parameters and so on),
``make_url`` builds a URL string from such an object or a dict. ``make_qs``, ``add_params``,
``extract_params`` and the ``quote_*`` functions handle query strings and URL quoting.

HTTP:

``http_request`` sends a request with ``requests``, using one session per process, so that
connections to the same host are reused. It does not raise on failures; instead, it returns an
``HTTPResponse`` with ``ok`` set to ``False``. Connection errors, timeouts and other request
errors get the pseudo status codes 900, 901 and 999. ``HTTPResponse.raise_if_failed`` turns a
failed response into an ``HTTPError`` subclass. The response text is decoded with the charset
from the content type header, or as UTF-8 or ISO-8859-1 if there is none.

Example::

    u = gws.lib.net.parse_url('https://example.com/wms?SERVICE=WMS')
    url = gws.lib.net.add_params(u.url, REQUEST='GetCapabilities')

    res = gws.lib.net.http_request(url, timeout=10)
    res.raise_if_failed()
    xml = res.text
"""

from typing import Optional

import re
import requests
import requests.adapters
import urllib.parse
import certifi

import gws
import gws.lib.osx


class Error(gws.Error):
    """Network error."""

    pass


class HTTPError(Error):
    """HTTP request failed."""

    pass


class Timeout(HTTPError):
    """HTTP request timed out."""

    pass


class ConnectionError(HTTPError):
    """HTTP connection failed."""

    pass


class GenericError(HTTPError):
    """HTTP request failed for another reason."""

    pass


_STATUS_CONNECTION_ERROR = 900
_STATUS_TIMEOUT = 901
_STATUS_GENERIC_ERROR = 999


class Url(gws.Data):
    """Components of a URL, as returned by ``parse_url``."""

    fragment: str
    """Fragment, without ``#``."""
    hostname: str
    """Host name."""
    netloc: str
    """Network location, including user name, password and port."""
    params: dict
    """Query parameters. For repeated parameters, the first value is used."""
    password: str
    """Password, unquoted."""
    path: str
    """Path."""
    pathparts: gws.lib.osx.ParsePathResult
    """Components of the path."""
    port: int
    """Port, ``0`` if not given."""
    qsl: list
    """Query parameters as a list of ``(name, value)`` pairs."""
    query: str
    """Query string, without ``?``."""
    scheme: str
    """Scheme, e.g. ``https``."""
    url: str
    """The parsed URL; with a ``//`` prefix if the original URL had no scheme."""
    username: str
    """User name, unquoted."""


def parse_url(url: str, **kwargs) -> Url:
    """Parse a URL.

    A URL without ``//`` is treated as starting with a host name.

    Args:
        url: URL string.
        **kwargs: Values to set on the ``Url`` object after parsing.

    Returns:
        A Url object.
    """

    if not is_abs_url(url):
        url = '//' + url

    us = urllib.parse.urlsplit(url)

    u = Url(
        fragment=us.fragment or '',
        hostname=us.hostname or '',
        netloc=us.netloc or '',
        params={},
        password=us.password or '',
        path=us.path or '',
        pathparts=gws.lib.osx.ParsePathResult(),
        port=0,
        qsl=[],
        query=us.query or '',
        scheme=us.scheme or '',
        url=url,
        username=us.username or '',
    )

    if us.port:
        try:
            u.port = int(us.port)
        except ValueError:
            pass

    if u.path:
        u.pathparts = gws.lib.osx.parse_path(u.path)

    if u.query:
        u.qsl = urllib.parse.parse_qsl(u.query)
        for k, v in u.qsl:
            u.params.setdefault(k, v)

    if u.username:
        u.username = unquote(u.username)
        u.password = unquote(u.get('password') or '')

    u.update(**kwargs)
    return u


_DEFAULT_PORTS = {'http': '80', 'https': '443'}


def make_url(u: Optional[Url | dict] = None, **kwargs) -> str:
    """Build a URL string.

    Uses ``scheme``, ``hostname``, ``port``, ``username``, ``password``, ``path``, ``params``
    and ``fragment``; other keys, like ``query`` or ``url``, are ignored. The port is omitted
    if it is the default port of the scheme.

    Args:
        u: URL components, a Url object or a dict.
        **kwargs: Components that override the values from ``u``.

    Returns:
        The URL.
    """

    p = gws.u.merge({}, u, kwargs)

    s = ''

    scheme = p.get('scheme', '').lower()
    if scheme:
        s += scheme + ':'

    host = p.get('hostname', '')
    port = p.get('port', '')
    path = p.get('path', '')

    if scheme or host:
        s += '//'

    if host:
        username = p.get('username', '')
        if username:
            s += quote_param(username) + ':' + quote_param(p.get('password', '')) + '@'

        s += host
        if port and str(port) != _DEFAULT_PORTS.get(scheme):
            s += ':' + str(port)

    if path:
        s += '/' + quote_path(path.lstrip('/'))

    params = p.get('params')
    if params:
        s += '?' + make_qs(params)

    fragment = p.get('fragment', '')
    if fragment:
        s += '#' + fragment.lstrip('#')

    return s


def parse_qs(x) -> dict:
    """Parse a query string.

    Args:
        x: Query string.

    Returns:
        A dict that maps each parameter name to a list of its values.
    """

    return urllib.parse.parse_qs(x)


def make_qs(x) -> str:
    """Convert a dict or a list of pairs to a query string.

    Values are encoded as UTF-8 and quoted. Booleans become ``true`` and ``false``,
    lists and other iterables are joined with commas, other values are converted with ``str``.

    Args:
        x: A dict, an object that can be converted to a dict, or a list of ``(name, value)`` pairs.

    Returns:
        The query string, without ``?``.
    """

    p = []
    items = x if isinstance(x, (list, tuple)) else gws.u.to_dict(x).items()

    def _value(v):
        if isinstance(v, (bytes, bytearray)):
            return v
        if isinstance(v, str):
            return v.encode('utf8')
        if v is True:
            return b'true'
        if v is False:
            return b'false'
        try:
            return b','.join(_value(y) for y in v)
        except TypeError:
            return str(v).encode('utf8')

    for k, v in items:
        k = urllib.parse.quote_from_bytes(_value(k))
        v = urllib.parse.quote_from_bytes(_value(v))
        p.append(k + '=' + v)

    return '&'.join(p)


def quote_param(s: str) -> str:
    """Quote a string for use in a URL, including slashes.

    Args:
        s: A string.

    Returns:
        The quoted string.
    """

    return urllib.parse.quote(s, safe='')


def quote_path(s: str) -> str:
    """Quote a URL path, leaving slashes as they are.

    Args:
        s: A path.

    Returns:
        The quoted path.
    """

    return urllib.parse.quote(s, safe='/')


def unquote(s: str) -> str:
    """Unquote a URL-quoted string.

    Args:
        s: A quoted string.

    Returns:
        The unquoted string.
    """

    return urllib.parse.unquote(s)


def add_params(url: str, params: dict = None, **kwargs) -> str:
    """Add query parameters to a URL.

    Existing parameters with the same names are replaced.

    Args:
        url: A URL.
        params: Parameters to add.
        **kwargs: More parameters to add.

    Returns:
        The new URL.
    """

    u = parse_url(url)
    if params:
        u.params.update(params)
    u.params.update(kwargs)
    return make_url(u)


def make_relative_url(path: str, params: dict = None, **kwargs) -> str:
    """Build a URL from a path and query parameters, without a scheme and host.

    Args:
        path: URL path; a leading slash is added if needed.
        params: Query parameters.
        **kwargs: More query parameters.

    Returns:
        The URL.
    """

    s = '/' + quote_path(path.lstrip('/'))
    p = {}
    if params:
        p.update(params)
    p.update(kwargs)
    if p:
        s += '?' + make_qs(p)
    return s


def extract_params(url: str) -> tuple[str, dict]:
    """Split the query parameters off a URL.

    Args:
        url: A URL.

    Returns:
        The URL without the query string, and the query parameters.
    """

    u = parse_url(url)
    params = u.params
    u.params = None
    return make_url(u), params


def is_abs_url(url):
    """Check if a URL has a host part, i.e. starts with ``//`` or a lowercase scheme and ``//``.

    Args:
        url: A URL.

    Returns:
        A truthy match object if the URL is absolute, ``None`` otherwise.
    """

    return re.match(r'^([a-z]+:|)//', url)


##


class HTTPResponse:
    """Response of ``http_request``.

    Attributes:
        ok: ``True`` if the request succeeded with a 2xx status.
        url: Request URL.
        content: Response body.
        content_type: Content type, without parameters.
        content_encoding: Charset from the content type header, or ``None``.
        status_code: HTTP status code, or a pseudo status code for requests that failed without a response.
    """

    def __init__(
        self,
        ok: bool,
        url: str,
        res: requests.Response = None,
        text: str = None,
        status_code=0,
    ):
        """Create a response.

        Args:
            ok: ``True`` if the request succeeded.
            url: Request URL.
            res: The ``requests`` response, if there is one.
            text: Content of a response without ``res``, e.g. an error message. It is stored as UTF-8 plain text.
            status_code: Status code of a response without ``res``.
        """
        self.ok = ok
        self.url = url
        if res is not None:
            self.content_type, self.content_encoding = _parse_content_type(res.headers)
            self.content = res.content
            self.status_code = res.status_code
        else:
            self.content_type, self.content_encoding = 'text/plain', 'utf8'
            self.content = text.encode('utf8') if text is not None else b''
            self.status_code = status_code

    @property
    def text(self) -> str:
        """Response body as text, decoded on first access.

        Returns:
            The decoded body.
        """

        if not hasattr(self, '_text'):
            setattr(self, '_text', _get_text(self.content, self.content_encoding))
        return getattr(self, '_text')

    def raise_if_failed(self):
        """Raise an error if the request failed.

        Raises:
            ``ConnectionError``: If the connection failed.
            ``Timeout``: If the request timed out.
            ``GenericError``: If the request failed for another reason, without a response.
            ``HTTPError``: If the response status is not 2xx.
        """

        if self.ok:
            return
        if self.status_code == _STATUS_CONNECTION_ERROR:
            raise ConnectionError(self.text)
        if self.status_code == _STATUS_TIMEOUT:
            raise Timeout(self.text)
        if self.status_code == _STATUS_GENERIC_ERROR:
            raise GenericError(self.text)
        raise HTTPError(f'HTTP error: {self.status_code}')


def _get_text(content, encoding) -> str:
    """Decode content with the given encoding, falling back to UTF-8 and ISO-8859-1."""

    if encoding:
        try:
            return str(content, encoding=encoding, errors='strict')
        except UnicodeDecodeError:
            pass

    # some folks serve utf8 content without a header, in which case requests thinks it's ISO-8859-1
    # (see http://docs.python-requests.org/en/master/user/advanced/#encodings)
    #
    # 'apparent_encoding' is not always reliable
    #
    # therefore when there's no header, we try utf8 first, and then ISO-8859-1

    try:
        return str(content, encoding='utf8', errors='strict')
    except UnicodeDecodeError:
        pass

    try:
        return str(content, encoding='ISO-8859-1', errors='strict')
    except UnicodeDecodeError:
        pass

    # both failed, do utf8 with replace

    gws.log.warning(f'decode failed')
    return str(content, encoding='utf8', errors='replace')


def _parse_content_type_header(header):
    """Split a content type header into the lowercase type and a dict of parameters."""

    parts = header.split(';')
    ctype = parts[0].strip().lower()
    params: dict[str, str] = {}

    for part in parts[1:]:
        part = part.strip()
        if '=' in part:
            k, v = part.split('=', 1)
            k = k.strip().lower()
            v = v.strip()
            # strip matched quotes (single or double)
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            params[k] = v

    return ctype, params


def _parse_content_type(headers):
    """Return the content type and the valid charset, or ``None``, from response headers."""

    # copied from requests.utils.get_encoding_from_headers, but with no ISO-8859-1 default

    header = headers.get('content-type')
    if not header:
        # https://www.w3.org/Protocols/rfc2616/rfc2616-sec7.html#sec7.2.1
        return 'application/octet-stream', None

    ctype, params = _parse_content_type_header(header)
    if 'charset' not in params:
        return ctype, None

    # make sure this is a valid python encoding
    enc = params['charset']
    try:
        str(b'.', encoding=enc, errors='strict')
    except LookupError:
        gws.log.warning(f'invalid content-type encoding {enc!r}')
        return ctype, None

    return ctype, enc


##

# @TODO locking for caches


def http_request(url, **kwargs) -> HTTPResponse:
    """Send an HTTP request.

    Failures are logged and returned as a response with ``ok`` set to ``False``, they do not raise.
    By default, the connect and read timeouts are 60 seconds, certificates are verified with ``certifi``,
    and a GBD WebSuite ``User-Agent`` header is sent.

    Args:
        url: Request URL.
        **kwargs: Options for ``requests.Session.request``. ``method`` is the HTTP method (``GET`` by default),
            ``params`` are added to the URL, a numeric ``timeout`` applies to both connect and read.

    Returns:
        The response.
    """

    kwargs = dict(kwargs)

    if 'params' in kwargs:
        url = add_params(url, kwargs.pop('params'))

    method = kwargs.pop('method', 'GET').upper()

    gws.debug.time_start(f'HTTP_{method}={url!r}')
    res = _http_request(method, url, kwargs)
    gws.debug.time_end()

    return res


_DEFAULT_CONNECT_TIMEOUT = 60
_DEFAULT_READ_TIMEOUT = 60

_USER_AGENT = f'GBD WebSuite (https://gbd-websuite.de)'

_POOL_SIZE = 16
"""Max. keep-alive connections per host and process."""

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Return the HTTP session of this process, creating it if needed."""
    # one session per process: reuses TCP/TLS connections to the same host across requests
    global _session
    if _session is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE)
        s.mount('http://', adapter)
        s.mount('https://', adapter)
        _session = s
    return _session


def _http_request(method, url, kwargs) -> HTTPResponse:
    """Send a request with default options and wrap the result in an ``HTTPResponse``."""

    kwargs['stream'] = False

    if 'verify' not in kwargs:
        kwargs['verify'] = certifi.where()

    timeout = kwargs.get('timeout', (_DEFAULT_CONNECT_TIMEOUT, _DEFAULT_READ_TIMEOUT))
    if isinstance(timeout, (int, float)):
        timeout = int(timeout), int(timeout)
    kwargs['timeout'] = timeout

    if 'headers' not in kwargs:
        kwargs['headers'] = {}
    kwargs['headers'].setdefault('User-Agent', _USER_AGENT)

    try:
        res = _get_session().request(method, url, **kwargs)
        if 200 <= res.status_code < 300:
            gws.log.debug(f'HTTP_OK_{method}: url={url!r} status={res.status_code!r}')
            return HTTPResponse(ok=True, url=url, res=res)
        gws.log.error(f'HTTP_FAILED_{method}: ({res.status_code!r}) url={url!r}')
        return HTTPResponse(ok=False, url=url, res=res)
    except requests.ConnectionError as exc:
        gws.log.error(f'HTTP_FAILED_{method}: (ConnectionError) url={url!r}')
        return HTTPResponse(ok=False, url=url, text=repr(exc), status_code=_STATUS_CONNECTION_ERROR)
    except requests.Timeout as exc:
        gws.log.error(f'HTTP_FAILED_{method}: (Timeout) url={url!r}')
        return HTTPResponse(ok=False, url=url, text=repr(exc), status_code=_STATUS_TIMEOUT)
    except requests.RequestException as exc:
        gws.log.error(f'HTTP_FAILED_{method}: (Generic: {exc!r}) url={url!r}')
        return HTTPResponse(ok=False, url=url, text=repr(exc), status_code=_STATUS_GENERIC_ERROR)

import cgi
import re
import requests
import requests.structures
import urllib.parse
import certifi

import gws
import gws.lib.osx
import gws.types as t


##

class Error(gws.Error):
    pass


class HTTPError(Error):
    pass


class Timeout(Error):
    pass


##

class Url(gws.Data):
    fragment: str
    hostname: str
    netloc: str
    params: dict
    password: str
    path: str
    pathparts: dict
    port: str
    qsl: list
    query: str
    scheme: str
    url: str
    username: str


def parse_url(url: str, **kwargs) -> Url:
    """Parse a string url and return an Url object"""

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
        pathparts={},
        port=str(us.port or ''),
        qsl=[],
        query=us.query or '',
        scheme=us.scheme or '',
        url=url,
        username=us.username or '',
    )

    if u.path:
        u.pathparts = gws.lib.osx.parse_path(u.path)

    if u.query:
        u.qsl = urllib.parse.parse_qsl(u.query)
        for k, v in u.qsl:
            u.params.setdefault(k.lower(), v)

    if u.username:
        u.username = unquote(u.username)
        u.password = unquote(u.get('password', ''))

    u.update(**kwargs)
    return u


def make_url(u: t.Optional[Url | dict] = None, **kwargs) -> str:
    p = gws.merge({}, u, kwargs)

    s = ''

    if p.get('scheme'):
        s += p['scheme'] + ':'

    s += '//'

    if p.get('username'):
        s += quote_param(p['username']) + ':' + quote_param(p.get('password', '')) + '@'

    if p.get('hostname'):
        s += p['hostname']
        if p.get('port'):
            s += ':' + str(p['port'])
        if p.get('path'):
            s += '/'
    else:
        s += '/'

    if p.get('path'):
        s += quote_path(p['path'].lstrip('/'))

    if p.get('params'):
        s += '?' + make_qs(p['params'])

    if p.get('fragment'):
        s += '#' + p['fragment'].lstrip('#')

    return s


def parse_qs(x) -> dict:
    return urllib.parse.parse_qs(x)


def make_qs(x) -> str:
    """Convert a dict/list to a query string.

    For each item in x, if it's a list, join it with a comma, stringify and in utf8.

    Args:
        x: Value, which can be a dict'able or a list of key,value pairs.

    Returns:
        The query string.
    """

    p = []
    items = x if isinstance(x, (list, tuple)) else gws.to_dict(x).items()

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
    return urllib.parse.quote(s, safe='')


def quote_path(s: str) -> str:
    return urllib.parse.quote(s, safe='/')


def unquote(s: str) -> str:
    return urllib.parse.unquote(s)


def add_params(url: str, params: dict = None, **kwargs) -> str:
    u = parse_url(url)
    if params:
        u.params.update(params)
    u.params.update(kwargs)
    return make_url(u)


def extract_params(url: str) -> tuple[str, dict]:
    u = parse_url(url)
    params = u.params
    u.params = None
    return make_url(u), params


def is_abs_url(url):
    return re.match(r'^([a-z]+:|)//', url)


##


class HTTPResponse:
    def __init__(self, ok: bool, url: str, res: requests.Response = None, text: str = None, status_code=0):
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
        if not hasattr(self, '_text'):
            setattr(self, '_text', _get_text(self.content, self.content_encoding))
        return getattr(self, '_text')

    def raise_if_failed(self):
        if not self.ok:
            raise HTTPError(self.status_code, self.text)


def _get_text(content, encoding) -> str:
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


def _parse_content_type(headers):
    # copied from requests.utils.get_encoding_from_headers, but with no ISO-8859-1 default

    content_type = headers.get('content-type')

    if not content_type:
        # https://www.w3.org/Protocols/rfc2616/rfc2616-sec7.html#sec7.2.1
        return 'application/octet-stream', None

    ctype, params = cgi.parse_header(content_type)
    if 'charset' not in params:
        return ctype, None

    enc = params['charset'].strip("'\"")

    # make sure this is a valid python encoding
    try:
        str(b'.', encoding=enc, errors='strict')
    except LookupError:
        gws.log.warning(f'invalid content-type encoding {enc!r}')
        return ctype, None

    return ctype, enc


##

# @TODO locking for caches


def http_request(url, **kwargs) -> HTTPResponse:
    kwargs = dict(kwargs)

    if 'params' in kwargs:
        url = add_params(url, kwargs.pop('params'))

    method = kwargs.pop('method', 'GET').upper()
    max_age = kwargs.pop('max_age', 0)
    cache_path = _cache_path(url)

    if method == 'GET' and max_age:
        age = gws.lib.osx.file_age(cache_path)
        if 0 <= age < max_age:
            gws.log.debug(f'HTTP_CACHED_{method}: url={url!r} path={cache_path!r} age={age}')
            return gws.unserialize_from_path(cache_path)

    gws.time_start(f'HTTP_{method}={url!r}')
    res = _http_request(method, url, kwargs)
    gws.time_end()

    if method == 'GET' and max_age and res.ok:
        gws.serialize_to_path(res, cache_path)

    return res


_DEFAULT_CONNECT_TIMEOUT = 60
_DEFAULT_READ_TIMEOUT = 60


def _http_request(method, url, kwargs) -> HTTPResponse:
    kwargs['stream'] = False

    if 'verify' not in kwargs:
        kwargs['verify'] = certifi.where()

    timeout = kwargs.get('timeout', (_DEFAULT_CONNECT_TIMEOUT, _DEFAULT_READ_TIMEOUT))
    if isinstance(timeout, (int, float)):
        timeout = int(timeout), int(timeout)
    kwargs['timeout'] = timeout

    try:
        res = requests.request(method, url, **kwargs)
        if 200 <= res.status_code < 300:
            gws.log.debug(f'HTTP_OK_{method}: url={url!r} status={res.status_code!r}')
            return HTTPResponse(ok=True, url=url, res=res)
        gws.log.error(f'HTTP_FAILED_{method}: ({res.status_code!r}) url={url!r}')
        return HTTPResponse(ok=False, url=url, res=res)
    except requests.Timeout as exc:
        gws.log.exception(f'HTTP_FAILED_{method}: (timeout) url={url!r}')
        return HTTPResponse(ok=False, url=url, text=repr(exc))
    except requests.RequestException as exc:
        gws.log.exception(f'HTTP_FAILED_{method}: ({exc!r}) url={url!r}')
        return HTTPResponse(ok=False, url=url, text=repr(exc))


def _cache_path(url):
    return gws.NET_CACHE_DIR + '/' + gws.sha256(url)

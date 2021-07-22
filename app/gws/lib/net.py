import cgi
import hashlib
import os
import pickle
import re
import time
import urllib.parse

import requests
import requests.structures

import gws
import gws.types as t

# https://urllib3.readthedocs.org/en/latest/security.html#using-your-system-s-root-certificates
CA_CERTS_PATH = '/etc/ssl/certs/ca-certificates.crt'


class Error(gws.Error):
    pass


class HTTPError(Error):
    pass


class Timeout(Error):
    pass


_parse_url_keys = (
    'dir',
    'ext',
    'filename',
    'fnbody',
    'fragment',
    'hostname',
    'netloc',
    'params',
    'password',
    'path',
    'port',
    'qs',
    'query',
    'scheme',
    'username',
)


def quote(s, safe='/'):
    return urllib.parse.quote(s, safe)


def unquote(s):
    return urllib.parse.unquote(s)


def is_abs_url(url):
    return re.match(r'^([a-z]+:|)//', url)


def parse_url(url):
    p = {k: '' for k in _parse_url_keys}

    # NB force an absolute url

    if not is_abs_url(url):
        url = '//' + url

    res = urllib.parse.urlsplit(url)

    for k in _parse_url_keys:
        p[k] = getattr(res, k, '') or ''

    if p['path']:
        p['dir'], p['filename'] = os.path.split(p['path'])
        if p['filename'].startswith('.'):
            p['fnbody'], p['ext'] = p['filename'], ''
        else:
            p['fnbody'], _, p['ext'] = p['filename'].partition('.')

    if p['query']:
        p['qs'] = urllib.parse.parse_qs(p['query'])
        r = {k: v[0] for k, v in p['qs'].items()}
    else:
        r = {}

    p['params'] = requests.structures.CaseInsensitiveDict(r)

    if p['username']:
        p['username'] = unquote(p['username'])
        p['password'] = unquote(p.get('password', ''))

    return p


def make_url(p):
    s = ''

    if p.get('scheme'):
        s += p['scheme']
        s += '://'
    else:
        s += '//'

    if p.get('username'):
        s += quote(p.get('username'))
        s += ':'
        s += quote(p.get('password', ''))
        s += '@'

    s += p['hostname']

    if p.get('port'):
        s += ':'
        s += str(p['port'])

    if p.get('path'):
        s += '/'
        s += p['path'].lstrip('/')

    if p.get('params'):
        s += '?'
        s += gws.as_query_string(dict(p['params']))

    if p.get('fragment'):
        s += '#'
        s += p['fragment'].lstrip('#')

    return s


def add_params(url, params):
    p = parse_url(url)
    p['params'].update(params)
    return make_url(p)


##


class HTTPResponse:
    def __init__(self, ok: bool, res: requests.Response = None, text: str = None, status_code=0):
        self._text = None
        self.ok = ok
        self.res = res
        if res:
            self.content_type, self.content_encoding = _parse_content_type(res.headers)
            self.content = res.content
            self.status_code = res.status_code
        else:
            self.content_type, self.content_encoding = 'text/plain', 'utf8'
            self.content = text.encode('utf8') if text is not None else b'???'
            self.status_code = status_code

    @property
    def text(self) -> str:
        if self._text is None:
            self._text = _get_text(self.content, self.content_encoding)
        return self._text


def _get_text(content, encoding):
    if encoding:
        try:
            return str(content, encoding=encoding, errors='strict')
        except UnicodeDecodeError:
            pass

    # some guys serve utf8 content without a header, in which case requests thinks it's ISO-8859-1
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

    gws.log.warn(f'decode failed')
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
        gws.log.warn(f'invalid content-type encoding {enc!r}')
        return ctype, None

    return ctype, enc


##

# @TODO locking for caches


def http_request(url, **kwargs) -> HTTPResponse:
    kwargs = dict(kwargs)

    max_age = int(kwargs.pop('max_age', 0))

    if max_age:
        cache_path = _cache_path(url)
        age = _file_age(cache_path)
        if age < max_age:
            gws.log.debug(f'REQUEST_CACHED: url={url!r} path={cache_path!r} age={age}')
            return _read_cache(cache_path)

    ts = gws.time_start(f'HTTP_REQUEST={url!r}')
    resp = _http_request(url, kwargs)
    gws.time_end(ts)

    if max_age and resp.ok:
        _store_cache(resp, _cache_path(url))

    return resp


_DEFAULT_CONNECT_TIMEOUT = 60
_DEFAULT_READ_TIMEOUT = 60


def _http_request(url, kwargs) -> HTTPResponse:
    if 'params' in kwargs:
        url = add_params(url, kwargs.pop('params'))

    kwargs['stream'] = False
    method = kwargs.pop('method', 'GET').upper()

    if 'verify' not in kwargs:
        kwargs['verify'] = CA_CERTS_PATH

    timeout = kwargs.get('timeout', (_DEFAULT_CONNECT_TIMEOUT, _DEFAULT_READ_TIMEOUT))
    if isinstance(timeout, (int, float)):
        timeout = int(timeout), int(timeout)
    kwargs['timeout'] = timeout

    try:
        res = requests.request(method, url, **kwargs)
        if res.status_code >= 400:
            gws.log.debug(f'HTTP_REQUEST_FAILED: (status) url={url!r} status={res.status_code!r}')
        return HTTPResponse(ok=(res.status_code < 400), res=res)
    except requests.Timeout as exc:
        gws.log.debug(f'HTTP_REQUEST_FAILED: (timeout) url={url!r}')
        return HTTPResponse(ok=False, text=repr(exc), status_code=500)
    except requests.RequestException as exc:
        gws.log.debug(f'HTTP_REQUEST_FAILED: (generic) url={url!r} err={exc!r}')
        return HTTPResponse(ok=False, text=repr(exc), status_code=500)


def _cache_path(url):
    return gws.NET_CACHE_DIR + '/' + gws.as_uid(url)


def _file_age(path):
    try:
        st = os.stat(path)
    except:
        return 1e20
    return int(time.time() - st.st_mtime)


def _store_cache(resp, path):
    gws.write_file_b(path, pickle.dumps(resp))


def _read_cache(path):
    return pickle.loads(gws.read_file_b(path))

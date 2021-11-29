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


# @TODO locking for caches


class Response:
    def __init__(self, resp: requests.Response):
        self.status_code = resp.status_code
        self.content = resp.content
        self.content_type, self.content_type_encoding = self._parse_content_type(resp.headers)
        self._text = None

    @property
    def text(self):
        if self._text is None:
            self._text = self._get_text()
        return self._text

    def _get_text(self):

        if self.content_type_encoding:
            try:
                return str(self.content, encoding=self.content_type_encoding, errors='strict')
            except UnicodeDecodeError:
                pass

        # some guys serve utf8 content without a header, in which case requests thinks it's ISO-8859-1
        # (see http://docs.python-requests.org/en/master/user/advanced/#encodings)
        #
        # 'apparent_encoding' is not always reliable
        #
        # therefore when there's no header, we try utf8 first, and then ISO-8859-1

        try:
            return str(self.content, encoding='utf8', errors='strict')
        except UnicodeDecodeError:
            pass

        try:
            return str(self.content, encoding='ISO-8859-1', errors='strict')
        except UnicodeDecodeError:
            pass

        # both failed, do utf8 with replace

        gws.log.warn(f'decode failed')
        return str(self.content, encoding='utf8', errors='replace')

    def _parse_content_type(self, headers):
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


class FailedResponse:
    def __init__(self, err):
        self.status_code = 500
        self.content = repr(err).encode('utf8')
        self.content_type = 'text/plain'
        self.content_type_encoding = 'utf8'
        self.text = repr(err)


def http_request(url, **kwargs) -> Response:
    if 'params' in kwargs:
        url = add_params(url, kwargs.pop('params'))
    cache_path = None
    max_age = kwargs.pop('max_age', 0)

    gws.log.debug(f'REQUEST_BEGIN: url={url!r} max_age={max_age}')

    if max_age:
        cache_path = _cache_path(url)
        ag = _file_age(cache_path)
        if ag < max_age:
            gws.log.debug(f'REQUEST_CACHED: path={cache_path!r} age={ag}')
            return _read_cache(cache_path)
        gws.log.debug('not_cached', cache_path, ag, max_age)

    kwargs = dict(kwargs or {})
    kwargs['stream'] = False

    method = kwargs.pop('method', 'GET').upper()

    if 'verify' not in kwargs:
        kwargs['verify'] = CA_CERTS_PATH

    timeout = kwargs.pop('timeout', None)
    if timeout != 0:
        # timeout=0 means no timeout
        # timeout=None means default
        if timeout is None:
            timeout = 60, 120
        elif isinstance(timeout, (int, float)):
            timeout = int(timeout), int(timeout)
        kwargs['timeout'] = timeout

    lax = kwargs.pop('lax', False)
    ts = time.time()

    err = None
    resp = None

    try:
        resp = requests.request(method, url, **kwargs)
    except requests.Timeout as e:
        gws.log.debug(f'REQUEST_FAILED: timeout url={url!r}')
        if cache_path:
            err = e
        else:
            raise Timeout() from e
    except requests.RequestException as e:
        gws.log.debug(f'REQUEST_FAILED: generic url={url!r}')
        if cache_path:
            err = e
        else:
            raise HTTPError(500, str(e)) from e

    if resp is not None and not lax:
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            gws.log.debug(f'REQUEST_FAILED: http url={url!r}')
            raise HTTPError(resp.status_code, resp.text)

    ts = time.time() - ts
    if resp and not err:
        gws.log.debug(f'REQUEST_DONE: code={resp.status_code} len={len(resp.content)} time={ts:.3f}')
        r = Response(resp)
        if cache_path:
            _store_cache(r, cache_path)
    else:
        gws.log.debug(f'REQUEST_DONE: resp=FAILED time={ts:.3f}')
        r = FailedResponse(err)

    return r


def _cache_path(url):
    return gws.NET_CACHE_DIR + '/' + _cache_key(url)


def _cache_key(url):
    m = re.search(r'^(https?://)(.+?)(\?.+)?$', url)
    if not m:
        return _hash(url)
    return gws.as_uid(m.group(2)) + '_' + _hash(m.group(3))


def _hash(s):
    return hashlib.md5(gws.as_bytes(s)).hexdigest()


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

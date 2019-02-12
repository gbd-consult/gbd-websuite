import cgi
import hashlib
import os
import pickle
import re
import time
import urllib.parse
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


class OgcServiceError(Error):
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


def parse_url(url):
    p = {k: '' for k in _parse_url_keys}

    # NB force an absolute url

    if not re.match(r'^([a-z]+:|)//', url):
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
        p['username'] = urllib.parse.unquote(p['username'])
        p['password'] = urllib.parse.unquote(p.get('password', ''))

    return p


def make_url(p):
    s = ''

    if p.get('scheme'):
        s += p['scheme']
        s += '://'
    else:
        s += '//'

    if p.get('username'):
        s += urllib.parse.quote(p.get('username'))
        s += ':'
        s += urllib.parse.quote(p.get('password', ''))
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


def http_request(url, **kwargs) -> requests.Response:
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

    if url.startswith('https') and 'verify' not in kwargs:
        kwargs['verify'] = CA_CERTS_PATH

    timeout = kwargs.pop('timeout', (60, 120))  # (connect, read)
    if isinstance(timeout, (int, float)):
        timeout = int(timeout), int(timeout)
    kwargs['timeout'] = timeout

    lax = kwargs.pop('lax', False)
    ts = time.time()

    try:
        resp = requests.request(method, url, **kwargs)
    except requests.Timeout as e:
        gws.log.debug(f'REQUEST_FAILED: timeout url={url!r}')
        raise Timeout() from e
    except requests.RequestException as e:
        gws.log.debug(f'REQUEST_FAILED: generic url={url!r}')
        raise HTTPError(500, str(e)) from e

    if not lax:
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            gws.log.debug(f'REQUEST_FAILED: http url={url!r}')
            raise HTTPError(resp.status_code, resp.text)

    # some guys serve utf8 content without a header,
    # in which case requests thinks it's ISO-8859-1
    # (see http://docs.python-requests.org/en/master/user/advanced/#encodings)
    #
    # 'apparent_encoding' is not always reliable
    #
    # therefore we just assume that when there's no headers, it's utf8
    # @TODO check if it really is!

    enc = _get_encoding_from_headers(resp.headers)
    resp.encoding = enc or 'utf-8'

    ts = time.time() - ts
    gws.log.debug(f'REQUEST_DONE: code={resp.status_code} len={len(resp.content)} time={ts:.3f}')

    if cache_path:
        _store_cache(resp, cache_path)

    return resp


_ogc_error_strings = '<ServiceException', '<ServerException', '<ows:ExceptionReport'


def ogc_request(url, params=None, max_age=0):
    """Query a raw service response"""

    params = params or {}

    # some guys accept only uppercase params
    # params = {k.upper(): v for k, v in params.items()}

    # the reason to use lax is that we want an exception text from the server
    # even if the status != 200

    resp = http_request(url, params=params, lax=True, max_age=max_age)
    status = resp.status_code
    text = resp.text

    if any(p in text for p in _ogc_error_strings):
        raise OgcServiceError(text.strip())

    if status != 200:
        resp.raise_for_status()

    return text


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
    with open(path, 'wb') as fp:
        pickle.dump(resp, fp)
    with open(path + '.txt', 'wt') as fp:
        fp.write(resp.text)


def _read_cache(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


# copied from requests.utils, but with no ISO-8859-1 default

def _get_encoding_from_headers(headers):
    content_type = headers.get('content-type')

    if not content_type:
        return None

    content_type, params = cgi.parse_header(content_type)

    if 'charset' in params:
        return params['charset'].strip("'\"")

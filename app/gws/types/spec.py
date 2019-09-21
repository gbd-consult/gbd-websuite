"""Validate values according to specs"""

import os
import re
import gws.tools.misc as misc
from .data import Data


class Error(Exception):
    pass


def _to_string(x):
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode('utf8')
    raise ValueError()


def _quote(x):
    return '"%s"' % x


class Validator:
    def __init__(self, types, strict=False):
        self.types = types
        self.keys = []
        self.path = ''
        self.strict = strict

        self.handlers = {
            'bool': self._bool,
            'dict': self._dict,
            'enum': self._enum,
            'float': self._float,
            'int': self._int,
            'list': self._list,
            'object': self._object,
            'str': self._str,
            'bytes': self._bytes,
            'tuple': self._tuple,
            'union': self._union,

            'gws.types.dirpath': self._dirpath,
            'gws.types.filepath': self._filepath,
            'gws.types.duration': self._duration,
            'gws.types.formatstr': self._formatstr,
            'gws.types.regex': self._regex,
            'gws.types.crsref': self._crsref,
            'gws.types.url': self._url,
        }

    def error(self, code, msg, val):
        val = repr(val)
        if len(val) > 600:
            val = val[:600] + '...'
        raise Error(code + ': ' + msg, self.path, '.'.join(str(k) for k in self.keys), val)

    def get(self, val, tname, path=''):
        self.path = path
        return self._get(val, tname)

    def _get(self, val, tname):
        if tname in self.handlers:
            # noinspection PyUnresolvedReferences
            return self.handlers[tname](val, None)
        #print('*** get', tname, val)
        t = self.types[tname]
        return self.handlers[t['type']](val, t)

    def _str(self, val, t):
        try:
            return _to_string(val)
        except:
            self.error('ERR_MUST_BE_STRING', 'must be a string', val)

    def _bytes(self, val, t):
        try:
            return bytes(val)
        except:
            self.error('ERR_MUST_BE_BYTES', 'must be a byte buffer', val)

    def _int(self, val, t):
        try:
            return int(val)
        except:
            self.error('ERR_MUST_BE_INT', 'must be an integer', val)

    def _float(self, val, t):
        try:
            return float(val)
        except:
            self.error('ERR_MUST_BE_FLOAT', 'must be a float', val)

    def _bool(self, val, t):
        try:
            return bool(val)
        except:
            self.error('ERR_MUST_BE_BOOL', 'must be true or false', val)

    def _dict(self, val, t):
        return self.ensure(val, dict)

    def _object(self, val, t):
        val = self.ensure(val, dict)
        res = {}

        for p in t['props']:
            name = p['name']
            self.keys.append(name)
            try:
                res[name] = self._value(p, val)
            finally:
                self.keys.pop()

        if self.strict:
            existing = set(p['name'] for p in t['props'])
            unknown = [
                _quote(k)
                for k in val
                if k not in existing and not k.startswith(('//', '#'))
            ]
            if len(unknown) == 1:
                self.error('ERR_UNKNOWN_PROP', 'unknown property: %s' % unknown[0], val)
            if len(unknown) >= 2:
                self.error('ERR_UNKNOWN_PROP', 'unknown properties: %s' % ', '.join(unknown), val)

        return Data(res)

    def _value(self, p, val):
        if not hasattr(val, 'get'):
            return self.error('ERR_UNEXPECTED', 'unexpected value', val)

        v2 = val.get(p['name'])
        df = p['default']

        # no value?

        if v2 is None:
            if not p['optional']:
                return self.error('ERR_MISSING_PROP', 'required property missing: %s' % p['name'], 'nothing')

            # no default as well
            if df is None:
                return None

            # the default, if given, must match the type
            # NB, for Data objects, default={} will create an objects with defaults
            return self._get(df, p['type'])

        # have value...

        if p['type'] == 'gws.types.literal':
            # if literal, the value must be exactly p[default]
            if v2 != df:
                return self.error('ERR_BAD_LITERAL', 'expected %s' % _quote(df), v2)
            return v2

        # no literal, normal case - the value is given and must match the type

        return self._get(v2, p['type'])

    def _union(self, val, t):
        if not hasattr(val, 'get'):
            return self.error('ERR_UNEXPECTED', 'unexpected value', val)

        tname = val.get('type')

        if tname:
            b = self._union_match(t['bases'], tname)
            if b:
                return self._get(val, b)

        return self.error('ERR_BAD_TYPE', 'invalid type', tname)


        # if 'type' in val:
        #     b = self._union_match(t['bases'], val['type'])
        #     if b:
        #         return self._get(val, b)
        #
        # err = []
        # for b in t['bases']:
        #     try:
        #         return self._get(val, b)
        #     except Error as e:
        #         err.append(e)
        #
        # raise err[0]

    def _union_match(self, bases, tname):
        for b in bases:
            p = b.split('.')
            if p[-2] == tname:
                return b

    def _list(self, val, t):
        val = self.ensure(val, list)
        res = []

        for n, v in enumerate(val):
            self.keys.append(n)
            try:
                res.append(self._get(v, t['base']))
            finally:
                self.keys.pop()

        return res

    def _tuple(self, val, t):
        val = self.ensure(val, list)
        if len(val) != len(t['bases']):
            self.error('ERR_BAD_TYPE', 'expected %s' % t['name'], val)

        res = []

        for n, v in enumerate(val):
            self.keys.append(n)
            try:
                res.append(self._get(v, t['bases'][n]))
            finally:
                self.keys.pop()

        return res

    def _enum(self, val, t):
        # NB: our Enums (see __init__) accept both names (for configs) and values (for api calls)
        # this blocks silly things like Enum {foo=bar bar=123} but we don't care
        for k, v in t['values'].items():
            if val == k or val == v:
                return v
        vals = ' or '.join(_quote(v) for v in t['values'])
        self.error('ERR_BAD_ENUM', 'invalid value (expected %s)' % vals, val)

    def _dirpath(self, val, t):
        path = os.path.join(os.path.dirname(self.path), val)
        if not os.path.isdir(path):
            self.error('ERR_DIR_NOT_FOUND', 'directory not found', path)
        return path

    def _filepath(self, val, t):
        path = os.path.join(os.path.dirname(self.path), val)
        if not os.path.isfile(path):
            self.error('ERR_FILE_NOT_FOUND', 'file not found', path)
        return path

    def _duration(self, val, t):
        try:
            return misc.parse_duration(val)
        except ValueError:
            self.error('ERR_BAD_DURATION', 'invalid duration', val)

    def _formatstr(self, val, t):
        return self._str(val, t)

    def _regex(self, val, t):
        try:
            re.compile(val)
            return val
        except re.error as e:
            self.error('ERR_BAD_REGEX', 'invalid regular expression: %s' % e, val)

    def _crsref(self, val, t):
        # @TODO: crs validation
        return self._str(val, t)

    def _url(self, val, t):
        # @TODO: url validation
        return self._str(val, t)

    def ensure(self, val, t):
        if isinstance(val, t):
            return val
        if t == list and isinstance(val, tuple):
            return list(val)
        if t == dict and isinstance(val, Data):
            return val.as_dict()
        if isinstance(t, type):
            t = 'object' if t == dict else t.__name__
        self.error('ERR_WRONG_TYPE', '%s expected' % t, val)

import re


class DecodeError(ValueError):
    pass


class Error:
    INVALID_TOKEN = 'invalid token'
    PREMATURE_EOF = 'premature end of input'
    DECODE_ERROR = 'string encoding error'
    INVALID_NUMBER = 'invalid number'
    INVALID_KEY = 'invalid key'
    ARRAY_EXPECTED = 'array expected'
    OBJECT_EXPECTED = 'object expected'


class Decoder:
    def __init__(self, opts):
        self.text = ''
        self.tokens = []
        self.tpos = 0
        self.opts = opts

    def decode(self, text):
        self.text = text

        self.tokens, p = _tokenize(text, _tokens)
        if p >= 0:
            self.error(Error.INVALID_TOKEN, p)

        self.skip(T_WS, T_COMMENT)

        if self.opts.get('as_object'):
            res = self.decode_as_object()
        elif self.opts.get('as_array'):
            res = self.decode_as_array()
        else:
            res = self.decode_value()

        self.skip(T_WS, T_COMMENT)

        if self.has_tok():
            t, s, p = self.tok()
            self.error(Error.INVALID_TOKEN, p)

        return res

    def decode_as_object(self):
        t, s, p = self.tok()
        if t == '{':
            self.pop()
            return self.decode_object(s, p)
        return self.decode_object(s, p, eof_ok=True)

    def decode_as_array(self):
        t, s, p = self.tok()
        if t == '[':
            self.pop()
            return self.decode_array(s, p)
        return self.decode_array(s, p, eof_ok=True)

    def decode_value(self):
        t, s, p = self.tok()

        if t == T_NULL:
            self.pop()
            return None

        if t == T_BOOL:
            self.pop()
            return s.lower() in ('true', 'on', 'yes', 'y')

        if t == T_NUMBER:
            self.pop()
            return self.decode_number(s, p)

        if t == T_HEXNUM:
            self.pop()
            return self.decode_number(s, p, hex=True)

        if t == T_STRD_1:
            self.pop()
            return self.decode_string(s[1:-1], p, rich=True, squeeze=True)
        if t == T_STRD_3:
            self.pop()
            return self.decode_string(s[3:-3], p, rich=True, dedent=True)
        if t == T_STRS_1:
            self.pop()
            return self.decode_string(s[1:-1], p, squeeze=True)
        if t == T_STRS_3:
            self.pop()
            return self.decode_string(s[3:-3], p, dedent=True)

        if t == T_RAW:
            self.pop()
            return s

        if t == '{':
            self.pop()
            return self.decode_object(s, p)

        if t == '[':
            self.pop()
            return self.decode_array(s, p)

        self.error(Error.INVALID_TOKEN, p)

    def decode_number(self, s, p, hex=False):
        s = s.replace('_', '')
        try:
            if hex:
                return int(s, 16)
            if '.' in s or 'e' in s or 'E' in s:
                return float(s)
            return int(s, 10)
        except ValueError:
            self.error(Error.INVALID_NUMBER, p)

    def decode_string(self, s, p, rich=False, squeeze=False, dedent=False):
        if '\n' in s:
            if squeeze:
                s = re.sub(r'\s+', ' ', s.strip())
            elif dedent:
                s = _dedent(s)

        if not rich:
            return s

        try:
            return _unescape(s)
        except (ValueError, IndexError, KeyError):
            # @TODO: report exact position of a string error
            self.error(Error.DECODE_ERROR, p)

    def decode_array(self, s, p, eof_ok=False):
        r = []

        while True:
            self.skip(T_WS, T_COMMENT, ',')

            if not self.has_tok() and eof_ok:
                return r

            t, s, p = self.tok()
            if t == ']':
                self.pop()
                return r

            r.append(self.decode_value())

    def decode_object(self, s, p, eof_ok=False):
        r = {}

        while True:
            self.skip(T_WS, T_COMMENT, ',')

            if not self.has_tok() and eof_ok:
                return r

            t, s, p = self.tok()
            if t == '}':
                self.pop()
                return r

            if t == T_RAW:
                self.pop()
                key = s
                raw_key = True
            else:
                key = self.decode_value()
                raw_key = False

            self.skip(T_WS, T_COMMENT, ':', '=')
            val = self.decode_value()

            if raw_key:
                self.store_raw(r, key, val, p)
            elif isinstance(key, (str, int, bool)):
                r[key] = val
            else:
                self.error(Error.INVALID_KEY, p)

    def store_raw(self, obj, key, val, p):
        o = {'r': obj}
        k = 'r'

        key = re.sub(r'\+([^.]+)', r'\1.+', key)
        key = key.split('.')

        for s in key:
            if s.isdigit():
                o = self.store_one(o, k, [], True, p)
                k = int(s)
            elif s == '+':
                o = self.store_one(o, k, [], True, p)
                k = -1
            else:
                o = self.store_one(o, k, {}, True, p)
                k = s
        self.store_one(o, k, val, False, p)

    def store_one(self, o, k, val, is_default, p):
        if isinstance(k, int):
            if not isinstance(o, list):
                return self.error(Error.ARRAY_EXPECTED, p)
            le = len(o)
            if k == -1:
                k = le
                o.append(None)
            elif k >= le:
                o.extend([None] * (k - le + 1))
            if not is_default:
                o[k] = val
            elif o[k] is None:
                o[k] = val
            return o[k]

        if not isinstance(o, dict):
            return self.error(Error.OBJECT_EXPECTED, p)
        if not is_default:
            o[k] = val
        elif k not in o:
            o[k] = val
        return o[k]

    def skip(self, *ts):
        while self.has_tok():
            t, s, p = self.tok()
            if t not in ts:
                return
            self.pop()

    def tok(self):
        try:
            t, s, p = self.tokens[self.tpos]
            if t == T_SPEC:
                return s, s, p
            return t, s, p
        except IndexError:
            self.error(Error.PREMATURE_EOF, len(self.text) - 1)

    def pop(self):
        self.tpos += 1

    def has_tok(self):
        return self.tpos < len(self.tokens)

    def error(self, msg, pos):
        r, c = _rowcol(self.text, pos)
        raise DecodeError(msg, r, c)


_rc = re.compile

T_STRD_1 = 2
T_STRD_3 = 3
T_STRS_1 = 4
T_STRS_3 = 5

T_BOOL = 10
T_NUMBER = 11
T_HEXNUM = 12
T_NULL = 13

T_SPEC = 22
T_RAW = 23
T_INDEX = 24
T_DOT = 25

T_WS = 200
T_COMMENT = 201

_tokens = [
    [T_WS, _rc(r"\s+")],
    [T_COMMENT, _rc(r"(?s)/\*.*?\*/")],
    [T_COMMENT, _rc(r"#.*")],
    [T_COMMENT, _rc(r"//.*")],
    [T_HEXNUM, _rc(r"(?i)0x[A-F0-9_]+")],
    [T_NUMBER, _rc(r"[+-]?[\d_]+(\.[\d_]+)?([eE][+-]?[\d_]+)?")],
    [T_STRD_3, _rc(r'(?s)"{3}(\\.|.)*?"{3}')],
    [T_STRD_1, _rc(r'(?s)"(\\.|[^"])*"')],
    [T_STRS_3, _rc(r"(?s)'{3}.*?'{3}")],
    [T_STRS_1, _rc(r"(?s)'[^']*'")],
    [T_BOOL, _rc(r"(?i)\b(true|false|yes|no|y|n|on|off)\b")],
    [T_NULL, _rc(r"\bnull\b")],
    [T_SPEC, _rc(r"[\[\]{},=:]")],
    [T_RAW, _rc(r"""[^\s\[\]{},=:"'#/]+""")],
]


def _tokenize(text, tokens):
    p = 0
    le = len(text)
    toks = []

    while p < le:
        for t, r in tokens:
            m = r.match(text, p)
            if m:
                s = m.group(0)
                toks.append((t, s, p))
                p += len(s)
                break
        else:
            return toks, p

    return toks, -1

def _rowcol(s, p):
    for row, ln in enumerate(s.splitlines(keepends=True), 1):
        if len(ln) > p:
            return row, p + 1
        p -= len(ln)
    return '?', '?'


def _dedent(s):
    minlen = 1e20
    buf = s.splitlines()

    if not buf[0].strip():
        buf.pop(0)
    if not buf[-1].strip():
        buf.pop(-1)

    for ln in buf:
        ls = ln.lstrip()
        if ls:
            minlen = min(minlen, len(ln) - len(ls))

    return '\n'.join(ln[minlen:].rstrip() for ln in buf)


def _remove_ws(s):
    return re.sub(r'\s+', ' ', s.strip())


# based on py_scanstring in json/decoder.py

_escapes = {
    '"': '"', '\\': '\\', '/': '/',
    'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t',
}


def _unescape(s):
    out = ''
    pos = 0

    while True:
        sp = s.find('\\', pos)
        if sp < 0:
            return out + s[pos:]

        out += s[pos:sp]
        e = s[sp + 1]

        if e == 'u':
            c, surrogate = _unescape_unicode_4(s, sp)
            out += c
            pos = sp + (12 if surrogate else 6)

        elif e == 'U':
            out += chr(_codepoint(s, sp + 2, 8))
            pos = sp + 10

        else:
            out += _escapes[e]
            pos = sp + 2


def _unescape_unicode_4(s, sp):
    uni = _codepoint(s, sp + 2, 4)

    if 0xd800 <= uni <= 0xdbff:
        if s[sp + 6:sp + 8] == '\\u':
            uni2 = _codepoint(s, sp + 8, 4)
            if 0xdc00 <= uni2 <= 0xdfff:
                uni = 0x10000 + (((uni - 0xd800) << 10) | (uni2 - 0xdc00))
                return chr(uni), True
        raise ValueError('invalid surrogate pair')

    return chr(uni), False


def _codepoint(s, pos, siz):
    esc = s[pos:pos + siz]
    if len(esc) == siz and esc[1] not in 'xX':
        return int(esc, 16)
    raise ValueError('invalid hex number')

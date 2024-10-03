"""slon - Simple Lightweight Object Notation.

Data exchange and configuration notation, similar to JSON.

See <https://github.com/gebrkn/slon>
"""

__version__ = '0.2.2'
__author__ = 'Georg Barikin <georg@merribithouse.net>'


def loads(text, as_array=False, as_object=False, hooks=None):
    b = _Buf(text, _prepare_hooks(hooks))

    _ws(b)

    if as_array:
        v = _value(b) if _chr(b) == '[' else _array(b, b.p, term=_EOF)
    elif as_object:
        v = _value(b) if _chr(b) == '{' else _object(b, b.p, term=_EOF)
    else:
        v = _value(b)

    _ws(b)

    if _chr(b) == _EOF:
        return v

    _err(b, 'TRAILING_GARBAGE')


parse = loads

###

_ERRORS = {
    'INVALID_ESCAPE_SEQUENCE': 'Invalid escape sequence',
    'INVALID_HOOK': 'Invalid hook name',
    'INVALID_KEY_TYPE': 'Invalid key type',
    'INVALID_NUMBER': 'Invalid number',
    'INVALID_TOKEN': 'Invalid token',
    'MISSING_DELIMITER': 'Whitespace or "," expected',
    'MISSING_KEY_DELIMITER': 'Whitespace or ":" expected',
    'MUST_BE_ARRAY': 'Unable to assign index (array expected)',
    'MUST_BE_OBJECT': 'Unable to assign index (object expected)',
    'TRAILING_GARBAGE': 'Trailing garbage',
    'UNTERMINATED_ARRAY': 'Unterminated array',
    'UNTERMINATED_CALL': '")" expected',
    'UNTERMINATED_COMMENT': 'Unterminated block comment',
    'UNTERMINATED_OBJECT': 'Unterminated object',
    'UNTERMINATED_STRING': 'Unterminated string',
}


class SlonError(ValueError):
    def __init__(self, code, message, pos, row, col, start_pos, start_row, start_col):
        ValueError.__init__(self, message, pos, row, col, start_pos, start_row, start_col)
        self.code = code
        self.message = message
        self.pos = pos
        self.row = row
        self.col = col
        self.start_pos = start_pos
        self.start_row = start_row
        self.start_col = start_col


###

_EOF = '\U0001FFFF'
_EOL = '\n'
_SQ = "'"
_DQ = '"'
_SLASH = "\\"
_BACKTICK = '`'

_P_WS = 1 << 1
_P_PUNCT = 1 << 2
_P_LIST_DELIM = 1 << 3
_P_KEY_DELIM = 1 << 4
_P_DEC = 1 << 5
_P_HEX = 1 << 6
_P_OCT = 1 << 7
_P_BIN = 1 << 8

_P_NON_WORD = _P_WS | _P_PUNCT

_CMAP = {
    ' ': _P_WS,
    '\n': _P_WS,
    '\r': _P_WS,
    '\t': _P_WS,
    '\f': _P_WS,

    '[': _P_PUNCT,
    ']': _P_PUNCT,
    '{': _P_PUNCT,
    '}': _P_PUNCT,
    '(': _P_PUNCT,
    ')': _P_PUNCT,
    ',': _P_PUNCT | _P_LIST_DELIM,
    '=': _P_PUNCT | _P_KEY_DELIM,
    ':': _P_PUNCT | _P_KEY_DELIM,
    '#': _P_PUNCT,
    '/': _P_PUNCT,

    _SLASH: _P_PUNCT,
    _SQ: _P_PUNCT,
    _DQ: _P_PUNCT,
    _BACKTICK: _P_PUNCT,

    '0': _P_HEX | _P_DEC | _P_OCT | _P_BIN,
    '1': _P_HEX | _P_DEC | _P_OCT | _P_BIN,
    '2': _P_HEX | _P_DEC | _P_OCT,
    '3': _P_HEX | _P_DEC | _P_OCT,
    '4': _P_HEX | _P_DEC | _P_OCT,
    '5': _P_HEX | _P_DEC | _P_OCT,
    '6': _P_HEX | _P_DEC | _P_OCT,
    '7': _P_HEX | _P_DEC | _P_OCT,
    '8': _P_HEX | _P_DEC,
    '9': _P_HEX | _P_DEC,
    'a': _P_HEX,
    'b': _P_HEX,
    'c': _P_HEX,
    'd': _P_HEX,
    'e': _P_HEX,
    'f': _P_HEX,
    'A': _P_HEX,
    'B': _P_HEX,
    'C': _P_HEX,
    'D': _P_HEX,
    'E': _P_HEX,
    'F': _P_HEX,

}

_ESCAPES = {
    "'": "'",
    '"': '"',
    '0': '\0',
    '/': '/',
    '\\': '\\',
    'b': '\b',
    'f': '\f',
    'n': '\n',
    'r': '\r',
    't': '\t',
}

_WORDS = {
    'true': True,
    'on': True,
    'yes': True,
    'false': False,
    'off': False,
    'no': False,
    'null': None,
    'none': None,
}

_SURR_1_START = 0xD800
_SURR_1_END = 0xDBFF
_SURR_2_START = 0xDC00
_SURR_2_END = 0xDFFF

_MAX_UNICODE = 0x110000


###

class _Buf:
    def __init__(self, text, hooks):
        self.text = text
        self.tlen = len(text)
        self.p = 0
        self.hooks = hooks


###

def _prepare_hooks(hooks):
    if not hooks:
        return {}

    # 'hooks' can be an iterable, e.g. a dict
    try:
        _ = "test" in hooks
        return hooks
    except TypeError:
        pass

    # ...or an object
    d = {}
    for key in dir(hooks):
        if not key.startswith('__'):
            f = getattr(hooks, key, None)
            if callable(f):
                d[key] = f
    return d


###

def _value(b):
    ch = _chr(b)

    if ch == _SQ:
        if _chr(b, 1) == _SQ and _chr(b, 2) == _SQ:
            b.p += 3
            return _single3(b, b.p - 3)
        else:
            b.p += 1
            return _single1(b, b.p - 1)

    if ch == _DQ:
        if _chr(b, 1) == _DQ and _chr(b, 2) == _DQ:
            b.p += 3
            return _double3(b, b.p - 3, backtick=False)
        else:
            b.p += 1
            return _double1(b, b.p - 1)

    if ch == _BACKTICK:
        b.p += 1
        return _double3(b, b.p - 1, backtick=True)

    if ch == '+':
        b.p += 1
        return _number(b, b.p - 1)

    if ch == '-':
        b.p += 1
        return -1 * _number(b, b.p - 1)

    if _CMAP.get(ch, 0) & _P_DEC or ch == '.':
        return _number(b, b.p)

    if ch == '[':
        b.p += 1
        return _array(b, b.p - 1, term=']')

    if ch == '{':
        b.p += 1
        return _object(b, b.p - 1, term='}')

    if _CMAP.get(ch, 0) & _P_NON_WORD:
        _err(b, 'INVALID_TOKEN')

    return _word(b, b.p)


###

def _single1(b, start):
    out = ''

    while 1:
        ch = _chr(b)
        if ch == _EOL or ch == _EOF:
            _err(b, 'UNTERMINATED_STRING', start)
        if ch == _SQ:
            b.p += 1
            break
        out += ch
        b.p += 1

    return out


def _single3(b, start):
    out = ''
    dedent = _CMAP.get(_chr(b), 0) & _P_WS

    while 1:
        ch = _chr(b)
        if ch == _EOF:
            _err(b, 'UNTERMINATED_STRING', start)
        if ch == _SQ and _chr(b, 1) == _SQ and _chr(b, 2) == _SQ:
            b.p += 3
            break
        out += ch
        b.p += 1

    out = _dedent(out) if dedent else _compress(out)
    return out


def _double1(b, start):
    out = ''

    while 1:
        ch = _chr(b)
        if ch == _EOL or ch == _EOF:
            _err(b, 'UNTERMINATED_STRING', start)
        if ch == _DQ:
            b.p += 1
            break
        if ch == _SLASH:
            b.p += 1
            out += _escape(b, b.p - 1)
        else:
            out += ch
            b.p += 1

    return out


def _double3(b, start, backtick):
    out = ''
    escapes = []
    esc_mark = _EOF
    dedent = _CMAP.get(_chr(b), 0) & _P_WS

    while 1:
        ch = _chr(b)
        if ch == _EOF:
            _err(b, 'UNTERMINATED_STRING', start)
        if backtick and ch == _BACKTICK:
            b.p += 1
            break
        if not backtick and ch == _DQ and _chr(b, 1) == _DQ and _chr(b, 2) == _DQ:
            b.p += 3
            break
        if ch == _SLASH:
            # escapes shouldn't affect the reflow,
            # so place them in a buffer and paste afterwards
            out += esc_mark
            b.p += 1
            escapes.append(_escape(b, b.p - 1))
        else:
            out += ch
            b.p += 1

    out = _dedent(out) if dedent else _compress(out)
    if not escapes:
        return out

    # paste escapes back

    out2 = ''
    n = 0
    for ch in out:
        if ch == esc_mark:
            out2 += escapes[n]
            n += 1
        else:
            out2 += ch
    return out2


def _dedent(s):
    indent = 100_000
    lines = [ln.rstrip() for ln in s.split(_EOL)]

    if lines and not lines[0]:
        lines.pop(0)
    if lines and not lines[-1]:
        lines.pop(-1)

    for ln in lines:
        if ln:
            indent = min(indent, len(ln) - len(ln.strip()))

    return _EOL.join(ln[indent:] for ln in lines)


def _compress(s):
    return ' '.join(s.strip().split())


def _escape(b, start):
    ch = _chr(b)

    if ch in _ESCAPES:
        b.p += 1
        return _ESCAPES[ch]

    cp = _unicode_escape(b)
    if cp < 0 or cp >= _MAX_UNICODE:
        _err(b, 'INVALID_ESCAPE_SEQUENCE', pos=start)

    return chr(cp)


def _unicode_escape(b):
    ch = _chr(b)

    # \xXX
    if ch == 'x':
        b.p += 1
        return _hexval(b, 2, 2)

    # \UXXXXXXXX
    if ch == 'U':
        b.p += 1
        return _hexval(b, 8, 8)

    # \u{XX...}
    if ch == 'u' and _chr(b, 1) == '{':
        b.p += 2
        cp = _hexval(b, 1, 8)
        if cp < 0 or _chr(b) != '}':
            return -1
        b.p += 1
        return cp

    # \uXXXX
    if ch == 'u':
        b.p += 1
        cp = _hexval(b, 4, 4)
        if cp < 0:
            return -1

        # this is a high surrogate, try to read a low surrogate and return the combined char
        if _SURR_1_START <= cp <= _SURR_1_END:
            savep = b.p

            if _chr(b) == _SLASH and _chr(b, 1) == 'u':
                b.p += 2
                cp2 = _hexval(b, 4, 4)
                if _SURR_2_START <= cp2 <= _SURR_2_END:
                    return 0x10000 + (((cp - _SURR_1_START) << 10) | (cp2 - _SURR_2_START))

            # not a valid surrogate pair, backtrack
            b.p = savep

        return cp

    return -1


def _hexval(b, minlen, maxlen):
    out = ''
    n = 0

    while n < maxlen:
        ch = _chr(b)
        if _CMAP.get(ch, 0) & _P_HEX:
            out += ch
            b.p += 1
            n += 1
        else:
            break

    if minlen <= n <= maxlen:
        return _int(out, 16)

    return -1


###

def _number(b, start):
    if _chr(b) == '0':
        ch = _chr(b, 1)
        if ch == 'x' or ch == 'X':
            b.p += 2
            return _nondec(b, start, _P_HEX, 16)
        if ch == 'o' or ch == 'O':
            b.p += 2
            return _nondec(b, start, _P_OCT, 8)
        if ch == 'b' or ch == 'B':
            b.p += 2
            return _nondec(b, start, _P_BIN, 2)

    return _decnum(b, start)


def _decnum(b, start):
    i = _intseq(b, _P_DEC)

    f = ''
    if _chr(b) == '.':
        b.p += 1
        f = _intseq(b, _P_DEC)
        if not f:
            _err(b, 'INVALID_NUMBER', pos=start)

    if not i and not f:
        _err(b, 'INVALID_NUMBER', pos=start)

    e = ''
    esign = ''

    ch = _chr(b)
    if ch == 'e' or ch == 'E':
        b.p += 1
        ch = _chr(b)
        if ch == '+' or ch == '-':
            esign = ch
            b.p += 1
        e = _intseq(b, _P_DEC)
        if not e:
            _err(b, 'INVALID_NUMBER', pos=start)

    if f or e:
        return _float(i, f, esign, e)

    return _int(i, 10)


def _nondec(b, start, prop, base):
    n = _intseq(b, prop)
    if not n:
        _err(b, 'INVALID_NUMBER', pos=start)
    return _int(n, base)


def _intseq(b, prop):
    out = ''

    while 1:
        ch = _chr(b)
        if _CMAP.get(ch, 0) & prop:
            out += ch
            b.p += 1
        elif ch == '_':
            b.p += 1
        else:
            break

    return out


###

def _array(b, start, term):
    out = []
    has_ws = True

    _ws(b, _P_WS | _P_LIST_DELIM)

    while 1:
        ch = _chr(b)

        if ch == term:
            b.p += 1
            break

        if ch == _EOF:
            _err(b, 'UNTERMINATED_ARRAY', start)

        if not has_ws:
            _err(b, 'MISSING_DELIMITER')

        out.append(_value(b))
        has_ws = _ws(b, _P_WS | _P_LIST_DELIM)

    return out


###

def _object(b, start, term):
    out = {}
    has_ws = True

    _ws(b, _P_WS | _P_LIST_DELIM)

    while 1:
        ch = _chr(b)

        if ch == term:
            b.p += 1
            break

        if ch == _EOF:
            _err(b, 'UNTERMINATED_OBJECT', start)

        if not has_ws:
            _err(b, 'MISSING_DELIMITER')

        key_pos = b.p
        is_quoted = ch == _SQ or ch == _DQ or ch == _BACKTICK
        key = _value(b)

        if not _is_number(key) and not _is_bool(key) and not _is_str(key):
            _err(b, 'INVALID_KEY_TYPE', pos=key_pos)

        if not _ws(b, _P_WS | _P_KEY_DELIM):
            _err(b, 'MISSING_KEY_DELIMITER')

        val = _value(b)

        if _is_str(key) and not is_quoted and ('.' in key or '+' in key):
            _store(b, out, key, val, key_pos)
        else:
            out[key] = val

        has_ws = _ws(b, _P_WS | _P_LIST_DELIM)

    return out


def _store(b, obj, cmp_key, val, key_pos):
    obj = [obj]
    key = 0
    is_int = True

    keys = ['']
    n = 0
    for ch in cmp_key:
        if ch == '.':
            keys.append('')
            n += 1
        elif ch == '+':
            keys.append('+')
            n += 1
        else:
            keys[n] += ch

    for k in keys:
        if k.isdigit():
            obj = _store_one(b, obj, key, is_int, [], False, key_pos)
            key = _int(k, 10)
            is_int = True
        elif k == '+':
            obj = _store_one(b, obj, key, is_int, [], False, key_pos)
            key = -1
            is_int = True
        else:
            obj = _store_one(b, obj, key, is_int, {}, False, key_pos)
            key = k
            is_int = False

    _store_one(b, obj, key, is_int, val, True, key_pos)


def _store_one(b, obj, key, is_int, val, force, key_pos):
    if is_int:
        if not _is_array(obj):
            _err(b, 'MUST_BE_ARRAY', pos=key_pos)

        le = len(obj)
        if key == -1:
            key = le
        while key >= le:
            obj.append(None)
            le += 1

        if force or obj[key] is None:
            obj[key] = val
        return obj[key]

    else:
        if not _is_object(obj):
            _err(b, 'MUST_BE_OBJECT', pos=key_pos)

        if force or key not in obj:
            obj[key] = val
        return obj[key]


###

def _word(b, start):
    w = ''

    while 1:
        ch = _chr(b)
        if ch == _EOF or (_CMAP.get(ch, 0) & _P_NON_WORD):
            break
        w += ch
        b.p += 1

    # keyword?

    k = w.lower()
    if k in _WORDS:
        return _WORDS[k]

    # hook?

    if _chr(b) == '(':
        call_pos = b.p
        b.p += 1

        _ws(b)
        val = _value(b)
        _ws(b)

        if _chr(b) != ')':
            _err(b, 'UNTERMINATED_CALL', start=call_pos)

        b.p += 1

        if w not in b.hooks:
            _err(b, 'INVALID_HOOK', pos=start)
        return b.hooks[w](val)

    # none of the above, simple string
    return w


###

def _ws(b, prop=_P_WS):
    start = b.p

    while 1:
        ch = _chr(b)
        if _CMAP.get(ch, 0) & prop:
            b.p += 1
        elif ch == '#':
            b.p += 1
            _line_comment(b, b.p - 1)
        elif ch == '/' and _chr(b, 1) == '/':
            b.p += 2
            _line_comment(b, b.p - 2)
        elif ch == '/' and _chr(b, 1) == '*':
            b.p += 2
            _block_comment(b, b.p - 2)
        else:
            break

    return b.p > start


def _line_comment(b, start):
    while 1:
        ch = _chr(b)
        if ch == _EOF:
            break
        if ch == _EOL:
            b.p += 1
            break
        b.p += 1


def _block_comment(b, start):
    while 1:
        ch = _chr(b)
        if ch == _EOF:
            _err(b, 'UNTERMINATED_COMMENT', start)
        if ch == '*' and _chr(b, 1) == '/':
            b.p += 2
            break
        b.p += 1


def _chr(b, d=0):
    try:
        return b.text[b.p + d]
    except IndexError:
        return _EOF


def _int(s, base):
    return int(s, base)


def _float(i, f, esign, e):
    return float((i or '0') + '.' + (f or '0') + 'E' + (esign or '') + (e or '0'))


def _is_number(x):
    return isinstance(x, (int, float))


def _is_str(x):
    return isinstance(x, str)


def _is_bool(x):
    return isinstance(x, bool)


def _is_array(x):
    return isinstance(x, list)


def _is_object(x):
    return isinstance(x, dict)


###

def _err(b, code, start=None, pos=None):
    if pos is None:
        pos = b.p

    message = _ERRORS[code]

    row, col = _rowcol(b, pos)

    message = '%s: line %d column %d (offset %d)' % (message, row, col, pos)

    if start is not None:
        start_row, start_col = _rowcol(b, start)
        message += ', started at line %d column %d (offset %d)' % (start_row, start_col, start)
    else:
        start = pos
        start_row = row
        start_col = col

    raise SlonError(code, message, pos, row, col, start, start_row, start_col)


def _rowcol(b, pos):
    n = r = c = 0
    while n < pos:
        if b.text[n] == _EOL:
            r += 1
            c = 0
        else:
            c += 1
        n += 1
    return r + 1, c + 1

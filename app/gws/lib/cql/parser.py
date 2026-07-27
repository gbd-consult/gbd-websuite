"""CQL2-Text parser. See the package documentation for the parse tree format.

Reference:
    - https://docs.ogc.org/is/21-065r2/21-065r2.html#cql2-bnf
"""

import re
import datetime


def parse(s: str):
    """Parse a CQL2-Text expression.

    Args:
        s: CQL2-Text expression.

    Returns:
        A parse tree.

    Raises:
        `ParseError` if the expression is invalid.
    """

    parser = _Parser()
    return parser.parse(s)


class ParseError(Exception):
    pass


class Node:
    AND = 'AND'
    ARRAY = 'ARRAY'
    BETWEEN = 'BETWEEN'
    BOOL = 'BOOL'
    DATE = 'DATE'
    FLOAT = 'FLOAT'
    FUNCTION = 'FUNCTION'
    IN = 'IN'
    INT = 'INT'
    IS_NULL = 'IS_NULL'
    LIKE = 'LIKE'
    NAME = 'NAME'
    NOT = 'NOT'
    NOT_BETWEEN = 'NOT_BETWEEN'
    NOT_IN = 'NOT_IN'
    NOT_LIKE = 'NOT_LIKE'
    NOT_NULL = 'NOT_NULL'
    OR = 'OR'
    STRING = 'STRING'
    TIMESTAMP = 'TIMESTAMP'
    USER_FUNCTION = 'USER_FUNCTION'
    WKT = 'WKT'


class C:
    LITERALS = {
        Node.ARRAY,
        Node.BOOL,
        Node.DATE,
        Node.FLOAT,
        Node.INT,
        Node.STRING,
        Node.TIMESTAMP,
        Node.WKT,
    }

    COMPARISON_OPERATORS = {'=', '<>', '!=', '<', '<=', '>', '>='}
    NOT_EQUAL_OPERATORS = {'<>', '!='}
    ADDITIVE_OPERATORS = {'+', '-'}
    MULTIPLICATIVE_OPERATORS = {'*', '/', '%'}
    POWER_OPERATORS = {'^'}

    OPERATORS = COMPARISON_OPERATORS | ADDITIVE_OPERATORS | MULTIPLICATIVE_OPERATORS | POWER_OPERATORS

    KEYWORDS = {
        'and',
        'or',
        'not',
        'is',
        'like',
        'between',
        'in',
    }

    PREDICATE_KEYWORDS = {
        'not',
        'is',
        'like',
        'between',
        'in',
    }

    WKT_KEYWORDS = {
        'point',
        'linestring',
        'polygon',
        'multipoint',
        'multilinestring',
        'multipolygon',
        'geometrycollection',
    }

    FUNCTIONS = {
        's_intersects': 2,
        's_contains': 2,
        's_crosses': 2,
        's_disjoint': 2,
        's_equals': 2,
        's_overlaps': 2,
        's_touches': 2,
        's_within': 2,
        't_after': 2,
        't_before': 2,
        't_contains': 2,
        't_disjoint': 2,
        't_during': 2,
        't_equals': 2,
        't_finishedby': 2,
        't_finishes': 2,
        't_intersects': 2,
        't_meets': 2,
        't_metby': 2,
        't_overlappedby': 2,
        't_overlaps': 2,
        't_startedby': 2,
        't_starts': 2,
        'a_contains': 2,
        'a_containedby': 2,
        'a_equals': 2,
        'a_overlaps': 2,
        'bbox': 4,
        'date': 1,
        'timestamp': 1,
        'interval': 2,
        'casei': 1,
        'accenti': 1,
    }

    ARRAY_FUNCTIONS = {
        'a_contains',
        'a_containedby',
        'a_equals',
        'a_overlaps',
    }

    PATTERN_FUNCTIONS = {
        'casei',
        'accenti',
    }

##


class _Token:
    def __init__(self, type: str, value, pos: int):
        self.type = type
        self.value = value
        self.index = pos

    def __repr__(self):
        return f'_Token({self.type}, {self.value!r}, {self.index})'


_TOKENS = [
    ('WHITESPACE', r'\s+'),
    ('TIMESTAMP', r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?'),
    ('DATE', r'\d{4}-\d{2}-\d{2}'),
    ('NUMBER', r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'),
    ('STRING', r"'(?:[^'\\]|\\.|'')*'"),
    ('QUOTED', r'"(?:[^"\\]|\\.|"")*"'),
    ('IDENT', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('', r'<=|>=|<>|!=|[()[\],.+\-*/%^<>=]'),
]


class _Parser:
    def __init__(self):
        self.tokens = []
        self.index = 0

    def parse(self, s: str):
        self.tokens = list(self.tokenize(s))
        self.tokens.append(_Token('EOF', None, len(s)))
        self.index = 0

        e = self.parse_boolean_expression()
        if self.tok().type != 'EOF':
            raise self.error(f'unexpected token')
        return e

    def error(self, message: str, pos=None):
        if pos is None:
            pos = self.tok().index
        return ParseError(f'Parse error: {message} ({pos})')

    def tokenize(self, s):
        pos = 0
        while pos < len(s):
            tok = None
            for typ, pattern in _TOKENS:
                r = re.compile(pattern)
                m = r.match(s, pos=pos)
                if m:
                    v = m.group(0)
                    tok = _Token(typ or v, v, pos)
                    pos = m.end()
                    break
            if tok is None:
                raise self.error(f'unexpected character', pos)
            if tok.type != 'WHITESPACE':
                yield tok

    def node(self, type: str, *args):
        return {'type': type, 'args': list(args)}

    ##

    def tok(self) -> _Token:
        if self.index < len(self.tokens):
            return self.tokens[self.index]
        return self.tokens[-1]

    def peek(self, n=1) -> _Token:
        if self.index + n < len(self.tokens):
            return self.tokens[self.index + n]
        return self.tokens[-1]

    def pop(self) -> _Token:
        tok = self.tok()
        self.index += 1
        return tok

    def expect(self, token_type: str) -> _Token:
        tok = self.tok()
        if tok.type != token_type:
            raise self.error(f'expected {token_type}, got {tok.type}')
        return self.pop()

    def is_a(self, token_type: str) -> bool:
        tok = self.tok()
        return tok.type == token_type

    def is_ident(self, value: str) -> bool:
        tok = self.tok()
        return tok.type == 'IDENT' and tok.value.upper() == value.upper()

    def expect_ident(self, value: str) -> _Token:
        tok = self.tok()
        if tok.type != 'IDENT' or tok.value.upper() != value.upper():
            raise self.error(f'expected {value}, got {tok}')
        return self.pop()

    ##

    def parse_boolean_expression(self):
        return self.parse_or_expression()

    def parse_or_expression(self):
        args = [self.parse_and_expression()]
        while self.is_ident('OR'):
            self.pop()
            args.append(self.parse_and_expression())
        return [Node.OR, *args] if len(args) > 1 else args[0]

    def parse_and_expression(self):
        args = [self.parse_not_expression()]
        while self.is_ident('AND'):
            self.pop()
            args.append(self.parse_not_expression())
        return [Node.AND, *args] if len(args) > 1 else args[0]

    def parse_not_expression(self):
        if self.is_ident('NOT'):
            self.pop()
            e = self.parse_not_expression()
            return [Node.NOT, e]
        return self.parse_primary_expression()

    def parse_primary_expression(self):
        if self.is_a('('):
            # a parenthesized group is a boolean expression, unless it turns out
            # to be an operand, like in "(a + b) * c = 1"
            index = self.index
            self.pop()
            e = self.parse_boolean_expression()
            self.expect(')')
            if not self.is_operand_follower():
                return e
            self.index = index
        return self.parse_predicate()

    def is_operand_follower(self):
        tok = self.tok()
        if tok.type in C.OPERATORS:
            return True
        return tok.type == 'IDENT' and tok.value.lower() in C.PREDICATE_KEYWORDS

    def parse_predicate(self):
        tok = self.tok()
        if tok.type == 'EOF':
            raise self.error('unexpected end of expression')

        e = self.parse_expression()

        if self.is_ident('NOT'):
            self.pop()
            if self.is_ident('LIKE'):
                return self.parse_like_predicate(e, True)
            if self.is_ident('BETWEEN'):
                return self.parse_between_predicate(e, True)
            if self.is_ident('IN'):
                return self.parse_in_predicate(e, True)
            raise self.error(f'unexpected {self.tok().type!r}')

        if self.is_ident('IS'):
            return self.parse_is_null_predicate(e)
        if self.is_ident('LIKE'):
            return self.parse_like_predicate(e, False)
        if self.is_ident('BETWEEN'):
            return self.parse_between_predicate(e, False)
        if self.is_ident('IN'):
            return self.parse_in_predicate(e, False)

        return self.parse_comparison_predicate(e)

    def parse_comparison_predicate(self, e):
        tok = self.tok()
        if tok.type in C.COMPARISON_OPERATORS:
            self.pop()
            b = self.parse_expression()
            op = '<>' if tok.type in C.NOT_EQUAL_OPERATORS else tok.type
            return [op, e, b]
        return e

    def parse_like_predicate(self, e, is_not):
        self.expect_ident('LIKE')
        pattern = self.parse_pattern_expression()
        return [Node.NOT_LIKE if is_not else Node.LIKE, e, pattern]

    def parse_pattern_expression(self):
        if self.is_a('IDENT') and self.tok().value.lower() in C.PATTERN_FUNCTIONS and self.peek().type == '(':
            return self.parse_postfix_expression()
        return self.parse_string_literal()

    def parse_between_predicate(self, e, is_not):
        self.expect_ident('BETWEEN')
        a = self.parse_expression()
        self.expect_ident('AND')
        b = self.parse_expression()
        return [Node.NOT_BETWEEN if is_not else Node.BETWEEN, e, a, b]

    def parse_in_predicate(self, e, is_not):
        self.expect_ident('IN')
        if self.is_a('('):
            self.pop()
            a = self.parse_list(')')
        elif self.is_a('['):
            self.pop()
            a = self.parse_list(']')
        else:
            raise self.error('expected ( or [ after IN')
        return [Node.NOT_IN if is_not else Node.IN, e, *a]

    def parse_is_null_predicate(self, e):
        self.expect_ident('IS')
        is_not = False
        if self.is_ident('NOT'):
            self.pop()
            is_not = True
        self.expect_ident('NULL')
        return [Node.NOT_NULL if is_not else Node.IS_NULL, e]

    def parse_expression(self):
        return self.parse_additive_expression()

    def parse_additive_expression(self):
        a = self.parse_multiplicative_expression()
        while self.tok().type in C.ADDITIVE_OPERATORS:
            op = self.pop().value
            b = self.parse_multiplicative_expression()
            a = [op, a, b]
        return a

    def parse_multiplicative_expression(self):
        a = self.parse_power_expression()
        while self.tok().type in C.MULTIPLICATIVE_OPERATORS:
            op = self.pop().value
            b = self.parse_power_expression()
            a = [op, a, b]
        return a

    def parse_power_expression(self):
        a = self.parse_unary_expression()
        while self.tok().type in C.POWER_OPERATORS:
            op = self.pop().value
            b = self.parse_unary_expression()
            a = [op, a, b]
        return a

    def parse_unary_expression(self):
        if self.tok().type in C.ADDITIVE_OPERATORS:
            op = self.pop().value
            e = self.parse_unary_expression()
            if op == '-':
                return ['-', e]
            return e
        return self.parse_postfix_expression()

    def parse_postfix_expression(self):
        pos = self.tok().index
        e = self.parse_atom()
        if self.is_a('('):
            self.pop()
            return self.parse_call(e, pos)
        return e

    def parse_call(self, head, pos):
        if head[0] != Node.NAME:
            raise self.error('invalid function name', pos)

        name = '.'.join(head[1:])
        key = name.lower()

        if key in C.ARRAY_FUNCTIONS:
            args = self.parse_array_argument_list()
        else:
            args = self.parse_list(')')

        if key not in C.FUNCTIONS:
            return [Node.USER_FUNCTION, name, *args]

        if len(args) != C.FUNCTIONS[key]:
            raise self.error(f'invalid number of arguments for {name!r}', pos)
        return [Node.FUNCTION, key, *args]

    def parse_atom(self):
        tok = self.tok()

        if tok.type == 'EOF':
            raise self.error('unexpected EOF')

        if self.is_a('('):
            self.pop()
            expr = self.parse_boolean_expression()
            self.expect(')')
            return expr

        if self.is_a('['):
            self.pop()
            return [Node.ARRAY, *self.parse_list(']')]
        if self.is_a('NUMBER'):
            return self.parse_number_literal()
        if self.is_a('STRING'):
            return self.parse_string_literal()
        if self.is_a('TIMESTAMP'):
            return self.parse_timestamp_literal()
        if self.is_a('DATE'):
            return self.parse_date_literal()
        if self.is_ident('TRUE') or self.is_ident('FALSE'):
            return [Node.BOOL, self.pop().value.upper() == 'TRUE']
        if self.is_a('IDENT') and self.tok().value.lower() in C.WKT_KEYWORDS:
            p = self.peek()
            if p.type == '(' or (p.type == 'IDENT' and p.value.upper() == 'Z'):
                return self.parse_geometry_literal()
        if self.is_a('IDENT') or self.is_a('QUOTED'):
            return self.parse_name()

        raise self.error(f'unexpected token: {tok.type}')

    def parse_name(self):
        parts = []

        while True:
            tok = self.tok()
            if tok.type == 'QUOTED':
                parts.append(self.unquote(self.pop().value))
            elif tok.type == 'IDENT':
                parts.append(self.pop().value)
            else:
                raise self.error(f'expected identifier')
            if self.is_a('.'):
                self.pop()
                continue
            break

        return [Node.NAME, *parts]

    def parse_number(self):
        val = self.expect('NUMBER').value
        if '.' in val or 'e' in val.lower():
            return float(val)
        return int(val)

    def parse_number_literal(self):
        val = self.parse_number()
        return [Node.FLOAT if isinstance(val, float) else Node.INT, val]

    def parse_string_literal(self):
        val = self.expect('STRING').value
        return [Node.STRING, self.unquote(val)]

    def parse_timestamp_literal(self):
        val = self.pop().value
        if val.endswith('Z'):
            val = val[:-1] + '+00:00'
        try:
            return [Node.TIMESTAMP, datetime.datetime.fromisoformat(val)]
        except ValueError:
            raise self.error(f'invalid timestamp')

    def parse_date_literal(self):
        val = self.pop().value
        try:
            year, month, day = val.split('-')
            return [Node.DATE, datetime.date(int(year), int(month), int(day))]
        except ValueError:
            raise self.error(f'invalid date')

    def parse_geometry_literal(self):
        parts = []
        parens = 0
        has_word = False

        while True:
            if self.is_a('IDENT'):
                if has_word:
                    parts.append(' ')
                parts.append(self.pop().value.upper())
                has_word = True
            elif self.is_a('NUMBER'):
                if has_word:
                    parts.append(' ')
                parts.append(str(self.parse_number()))
                has_word = True
            elif self.is_a(','):
                self.pop()
                parts.append(', ')
                has_word = False
            elif self.is_a('('):
                self.pop()
                parts.append('(')
                has_word = False
                parens += 1
            elif self.is_a(')'):
                self.pop()
                parts.append(')')
                has_word = False
                parens -= 1
                if parens == 0:
                    break
            else:
                break

        return [Node.WKT, ''.join(parts)]

    def parse_array_argument_list(self):
        elements = []
        if not self.is_a(')'):
            elements.append(self.parse_array_argument())
            while self.is_a(','):
                self.pop()
                elements.append(self.parse_array_argument())
        self.expect(')')
        return elements

    def parse_array_argument(self):
        if self.is_a('('):
            self.pop()
            return [Node.ARRAY, *self.parse_list(')')]
        return self.parse_expression()

    def parse_list(self, end):
        elements = []
        if not self.is_a(end):
            elements.append(self.parse_expression())
            while self.is_a(','):
                self.pop()
                elements.append(self.parse_expression())
        self.expect(end)
        return elements

    def unquote(self, s: str):
        quote = s[0]
        return s[1:-1].replace('\\' + quote, quote).replace(quote + quote, quote)

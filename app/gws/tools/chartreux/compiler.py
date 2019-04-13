"""Template compiler"""

import ast
import os
import re


class Error(ValueError):
    def __init__(self, code, message, path, line):
        msg = _f('{}: {} in {} on line {}', code, message, repr(path), line)
        super().__init__(msg)
        self.code = code
        self.message = msg
        self.path = path
        self.line = line


ERROR_SYNTAX = 'syntax error'
ERROR_COMMAND = 'unknown command'
ERROR_IDENT = 'invalid identifier'
ERROR_EOF = 'unexpected end of file'
ERROR_DEF = 'function definition error'
ERROR_FILE = 'file not found'
ERROR_NOT_SUPPORTED = 'syntax not supported'
ERROR_ARG_NOT_SUPPORTED = 'argument syntax not supported'
ERROR_FILTER = 'invalid or unknown filter'

_DEFAULT_OPTIONS = {
    'filter': None,
    'globals': [],
    'name': 'render',
    'path': '',
    'silent': False,
}


##

def compile(text, **options):
    fname = '_RENDER'
    options['name'] = fname

    cc = Compiler(options)
    python = cc.run(text)

    try:
        locs = {}
        exec(python, {}, locs)
        return locs[fname]
    except SyntaxError as e:
        loc = _source_location(python, e.lineno - 1)
        _error(ERROR_SYNTAX, loc[0], loc[1], e.msg)


def compile_path(path, **options):
    options['path'] = path
    return compile(_read(path), **options)


def translate(text, **options):
    cc = Compiler(options)
    python = cc.run(text)

    try:
        exec(python, {})
    except SyntaxError as e:
        loc = _source_location(python, e.lineno - 1)
        _error(ERROR_SYNTAX, loc[0], loc[1], e.msg)

    return python


def translate_path(path, **options):
    options['path'] = path
    return translate(_read(path), **options)


##

class Source:
    def __init__(self, text, path):
        self.lines = text.splitlines(keepends=True) or ['']
        self.maxline = len(self.lines)
        self.lineno = 0
        self.path = path

    def next_line(self):
        if self.lineno >= self.maxline:
            return None
        self.lineno += 1
        return self.lines[self.lineno - 1]


class Parser:
    def __init__(self, compiler):
        self.cc = compiler
        self.source = []

    def add_source(self, text, path):
        self.source.append(Source(text, path))

    @property
    def current_source(self):
        return self.source[-1]

    def read_line(self):
        while True:
            ln = self.current_source.next_line()

            if ln is not None:
                return ln

            if len(self.source) > 1:
                self.source.pop()
                continue

            return None

    command_symbol = '@'
    comment_symbol = '##'

    command_re = r'^(\w+)(.*)'

    text_command = '\x00'

    def parse_line(self, ln):
        if ln is None:
            return 'eof', ''

        sl = ln.strip()

        if sl.startswith(self.comment_symbol):
            return None, ''

        if not sl.startswith(self.command_symbol):
            return self.text_command, ln

        m = re.search(self.command_re, sl[1:])
        if not m:
            return self.text_command, ln

        cmd = m.group(1).lower()

        # NB: the final : in commands is optional
        arg = m.group(2).lstrip().rstrip(' :')
        return cmd, arg

    def parse_until(self, *terminators):
        while True:
            ln = self.read_line()
            cmd, arg = self.parse_line(ln)

            if not cmd:
                continue

            if cmd == 'eof' and 'eof' not in terminators:
                self.cc.error(ERROR_EOF)

            if cmd in terminators:
                return cmd, arg

            self.cc.command.process(cmd, arg)


class Expression:
    _operators = {
        'Add': '+',
        'Div': '/',
        'FloorDiv': '//',
        'Mod': '%',
        'Mult': '*',
        'Pow': '**',
        'Sub': '-',
        'Not': 'not',
        'UAdd': '+',
        'USub': '-',
        'And': 'and',
        'Or': 'or',
        'Gt': '>',
        'GtE': '>=',
        'Lt': '<',
        'LtE': '<=',
        'Eq': '==',
        'NotEq': '!=',
    }

    def __init__(self, compiler):
        self.cc = compiler
        self.default_filter = [None, None]

    def _ast(self, arg):
        try:
            return ast.parse(arg.strip(), mode='eval')
        except SyntaxError:
            return self.cc.error(ERROR_SYNTAX)

    def get_default_filter(self):
        f = self.cc.option('filter')
        if not f:
            return

        if f != self.default_filter[0]:
            n = self._ast(f)
            self.default_filter = [f, n.body if _cname(n) == 'Expression' else n]

        return self.default_filter[1]

    def parse(self, arg, with_default_filter=False):
        n = self._ast(arg)
        has_filter = _cname(n.body) == 'BinOp' and _cname(n.body.op) == 'BitOr'
        flt = self.get_default_filter()

        if with_default_filter and flt and not has_filter:
            return self.make_filter(n, flt)

        return self.walk(n)

    def parse_ast(self, node):
        return self.walk(node)

    def parse_args(self, args):
        return self.walk_args(self.parse_args_ast(args))

    def parse_args_ast(self, args):
        return self._ast('_' + self.fix_args(args)).body

    def walk(self, n):
        t = _cname(n)

        if t == 'NameConstant':
            return repr(n.value)
        if t == 'Num':
            return repr(n.n)
        if t == 'Str':
            return repr(n.s)

        if t == 'List':
            return '[' + _comma(self.walk(x) for x in n.elts) + ']'
        if t == 'Dict':
            return '{' + _comma(_f('{}: {}', self.walk(k), self.walk(v)) for k, v in zip(n.keys, n.values)) + '}'

        if t == 'Expression':
            return self.walk(n.body)

        if t == 'Attribute':
            if self.is_global(n.value):
                return _f('{}.{}', self.walk(n.value), n.attr)
            return self.make_getter(self.walk(n.value), n.attr)

        if t == 'Subscript':
            return _f('{}[{}]', self.walk(n.value), self.walk(n.slice))
        if t == 'Index':
            return self.walk(n.value)

        if t == 'Name':
            if self.is_global(n) or self.is_local(n):
                return n.id
            return self.make_context_getter(n.id)

        if t == 'Call':
            return _f('{}({})', self.walk(n.func), _comma(self.walk_args(n)))
        if t == 'keyword':
            e = self.walk(n.value)
            if n.arg:
                return _f('{}={}', n.arg, e)
            return '**' + e
        if t == 'Starred':
            return _f('*{}', self.walk(n.value))

        if t == 'BinOp':
            if _cname(n.op) == 'BitOr':
                return self.make_filter(n.left, n.right)
            return _f('({}) {} ({})', self.walk(n.left), self.operator(n.op), self.walk(n.right))

        if t == 'BoolOp':
            o = _f(' {} ', self.operator(n.op))
            return o.join(_f('({})', self.walk(x)) for x in n.values)

        if t == 'UnaryOp':
            return _f('{} ({})', self.operator(n.op), self.walk(n.operand))

        if t == 'Compare':
            e = self.walk(n.left)
            for o, c in zip(n.ops, n.comparators):
                e += _f(' {} ({})', self.operator(o), self.walk(c))
            return e

        return self.cc.error(ERROR_NOT_SUPPORTED, t)

    def fix_args(self, args):
        args = args.strip()

        # @def foo

        if not args:
            return '()'

        # @def foo a b c

        if re.match(r'^[\w ]+$', args):
            args = args.split()
            return _f('({})', _comma(args))

        # @def foo a,b,c
        # @def foo (a,b,c)

        if not args.startswith('('):
            return '(' + args + ')'

        return args

    def walk_args(self, n):
        return [self.walk(a) for a in n.args] + [self.walk(a) for a in n.keywords]

    filter_prefix = 'filter_'

    def filter_call(self, name, args):
        # a filter is either a local var or 'self.filter_xxx'
        if name in self.cc.scope:
            return _f('{}({})', name, args)
        return _f('_RT.{}{}({})', self.filter_prefix, name, args)

    def make_filter(self, le, ri):
        t = _cname(ri)

        # xyz | html
        if t == 'Name':
            return self.filter_call(ri.id, self.walk(le))

        # xyz | cut(5)
        if t == 'Call':
            func = ri.func
            if _cname(func) != 'Name':
                return self.cc.error(ERROR_FILTER, t)
            args = [self.walk(le)] + self.walk_args(ri)
            return self.filter_call(func.id, _comma(args))

        # xyz | '{:1f}'
        if t == 'Str':
            fmt = ri.s
            if '{' not in fmt:
                fmt = '{' + fmt + '}'
            args = [self.walk(le), repr(fmt)]
            return self.filter_call('format', _comma(args))

        return self.cc.error(ERROR_FILTER, t)

    def operator(self, n):
        t = _cname(n)
        if t not in self._operators:
            return self.cc.error(ERROR_NOT_SUPPORTED, t)
        return self._operators[t]

    def is_global(self, n):
        t = _cname(n)
        if t == 'Attribute':
            return self.is_global(n.value)
        if t == 'Subscript':
            return self.is_global(n.value)
        if t == 'Name':
            return n.id in self.cc.globals
        return False

    def is_local(self, n):
        t = _cname(n)
        if t == 'Attribute':
            return self.is_local(n.value)
        if t == 'Subscript':
            return self.is_local(n.value)
        if t == 'Name':
            return n.id in self.cc.scope
        return False

    def make_getter(self, var, prop):
        return _f('_GET({},{})', var, repr(prop))

    def make_context_getter(self, var):
        self.cc.code.add_context_var(var)
        return _f('_GETCONTEXT({})', repr(var))


class Command:
    def __init__(self, compiler):
        self.cc = compiler

    def process(self, cmd, arg):
        # is this a chunk of text?

        if cmd == self.cc.parser.text_command:
            return self.text_command(arg)

        # do we have a block with this name?

        if cmd in self.cc.blocks:
            return self.user_block_command(cmd, arg)

        # scoped function name => line command

        if cmd in self.cc.scope:
            return self.user_line_command(cmd, arg)

        # built-in command

        handler = getattr(self, 'command_' + cmd, None)
        if handler:
            handler(arg)
            return

        # nothing found :(

        self.cc.error(ERROR_COMMAND, cmd)

    #

    def user_block_command(self, cmd, arg):
        v = self.cc.new_var()
        self.cc.code.add('_PUSHBUF()')
        self.cc.parser.parse_until('end')
        self.cc.code.add(_f('{} = _POPBUF()', v))
        args = self.cc.expression.parse_args(arg)
        args.insert(0, v)
        self.cc.code.try_block(_f('print({}({}))', cmd, _comma(args)))

    def user_line_command(self, cmd, arg):
        args = self.cc.expression.parse_args(arg)
        self.cc.code.try_block(_f('print({}({}))', cmd, _comma(args)))

    interpolation_re = r'''(?x) 
        { (?=\S)
            (
                " (\\. | [^"])* "
                |
                ' (\\. | [^'])* '
                |
                [^'"{}]
            )+
        } 
    '''

    def text_command(self, arg):
        # just a chunk of text, possibly with interpolation
        s = arg

        for m, val in _findany(self.interpolation_re, s):
            if m:
                e = self.cc.expression.parse(val[1:-1], with_default_filter=True)
                self.cc.code.try_block(_f('print({})', e))
            else:
                self.cc.code.string(val)

    # built-in commands

    def command_option(self, arg):
        # @option option value

        arg = arg.split()
        val = 'true' if len(arg) == 1 else arg[1]

        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False

        self.cc.options[arg[0]] = val

    def if_head(self, arg):
        v = self.cc.new_var()
        e = self.cc.expression.parse(arg)
        self.cc.code.try_block(
            _f('{} = {}', v, e),
            _f('{} = None', v)
        )
        self.cc.code.add(_f('if {}:', v))

    def command_if(self, arg):
        # @if cond ...flow... @elif cond ...flow... @else ...flow... @end

        """
            we need 'if' conditions to be statements, so we transform this

                if abc
                    xxxx
                elif def
                    yyyy
                else
                    zzzz

            into this

                for (once):
                    x = abc
                    if x:
                        xxxx
                        break

                    x = def
                    if x:
                        yyyy
                        break

                    if True:
                        zzzz
                        break


        """

        self.cc.code.add(_f('for {} in "_":', self.cc.new_var()))
        self.cc.code.begin()

        self.if_head(arg)
        self.cc.code.begin()

        while True:
            cmd, arg = self.cc.parser.parse_until('end', 'else', 'elif')
            if cmd == 'elif':
                self.cc.code.add('break')
                self.cc.code.end()
                self.if_head(arg)
                self.cc.code.begin()
                continue
            if cmd == 'end':
                self.cc.code.add('break')
                self.cc.code.end()
                break
            if cmd == 'else':
                self.cc.code.add('break')
                self.cc.code.end()
                self.cc.code.add('if True:')
                self.cc.code.begin()
                self.cc.parser.parse_until('end')
                self.cc.code.add('break')
                self.cc.code.end()
                break

        self.cc.code.end()

    def command_return(self, arg):
        # @return expr

        e = self.cc.expression.parse(arg)
        self.cc.code.add(_f('return {}', e))

    def parse_def_args(self, args):
        # NB: although this is formally a function def, we parse it as a call
        node = self.cc.expression.parse_args_ast(args)

        names = []
        signature = []

        for a in node.args:
            t = _cname(a)
            if t == 'Name':
                names.append(a.id)
                signature.append(a.id)
            elif t == 'Starred':
                names.append(a.value.id)
                signature.append(_f('*{}', a.value.id))
            else:
                return self.cc.error(ERROR_ARG_NOT_SUPPORTED, t)

        for a in node.keywords:
            t = _cname(a)
            if t == 'keyword':
                if a.arg:
                    names.append(a.arg)
                    signature.append(_f('{}={}', a.arg, self.cc.expression.parse_ast(a.value)))
                else:
                    names.append(a.value.id)
                    signature.append(_f('**{}', a.value.id))

            else:
                return self.cc.error(ERROR_ARG_NOT_SUPPORTED, t)

        return names, _comma(signature)

    func_def_re = r'^(\w+)(.*)$'

    def parse_def_body(self, arg):
        m = re.search(self.func_def_re, arg)
        if not m:
            self.cc.error(ERROR_DEF)

        name, args = m.groups()
        self.cc.scope.add(name)
        self.cc.frames.append(self.cc.scope)
        self.cc.scope = set(self.cc.scope)

        arg_names, signature = self.parse_def_args(args)
        self.cc.scope.update(arg_names)

        fun = self.cc.new_var()
        res = self.cc.new_var()
        buf = self.cc.new_var()

        # def's are wrapped in push/popbuf to allow explicit @return's
        #
        # def userfunc(args):
        #     def fun()
        #         ...
        #         [@return ...]
        #         ...
        #
        #     PUSHBUF
        #     res = fun()
        #     buf = POPBUF
        #     return buf if res is None else res
        #

        self.cc.code.add(_f('def {}({}):', name, signature))
        self.cc.code.begin()

        self.cc.code.add(_f('def {}():', fun))
        self.cc.code.begin()

        self.cc.parser.parse_until('end')
        self.cc.code.add('pass')
        self.cc.code.end()

        self.cc.code.add('_PUSHBUF()')
        self.cc.code.add(_f('{} = {}()', res, fun))
        self.cc.code.add(_f('{} = _POPBUF()', buf))
        self.cc.code.add(_f('return {} if {} is None else {}', buf, res, res))
        self.cc.code.end()

        self.cc.scope = self.cc.frames.pop()

        return name

    def command_def(self, arg):
        # @def name args ...body... @end

        self.parse_def_body(arg)

    def command_block(self, arg):
        # @block name args ...body... @end

        name = self.parse_def_body(arg)
        self.cc.blocks.add(name)

    def command_code(self, arg):
        # @code python-expression
        # @code ...python-code... @end

        if arg:
            buf = [arg]
        else:
            buf = []
            while True:
                ln = self.cc.parser.read_line()
                cmd, _ = self.cc.parser.parse_line(ln)
                if cmd == 'end':
                    break
                if cmd == 'eof':
                    return self.cc.error(ERROR_EOF)
                buf.append(ln.rstrip())

        self.cc.code.try_block(_reindent(buf), 'pass')

    def command_quote(self, arg):
        # @quote label ...flow... @end label

        label = arg
        while True:
            ln = self.cc.parser.read_line()
            cmd, arg = self.cc.parser.parse_line(ln)
            if cmd == 'end' and arg == label:
                break
            self.cc.code.string(ln)

    let_re = r'^(\w+)(.*)$'

    def command_let(self, arg):
        # @let var = expression ('=' is optional)
        # @let var ...flow... @end

        m = re.search(self.let_re, arg)
        if not m:
            return self.cc.error(ERROR_IDENT)

        name, expr = m.groups()
        expr = expr.strip()
        self.cc.scope.add(name)

        if expr:
            if expr.startswith('='):
                expr = expr[1:]
            e = self.cc.expression.parse(expr)
            self.cc.code.add(_f('{} = {}', name, e))
        else:
            self.cc.code.add('_PUSHBUF()')
            self.cc.parser.parse_until('end')
            self.cc.code.add(_f('{} = _POPBUF()', name))

    def command_var(self, arg):
        # @var var, var,...

        names = [x.strip() for x in arg.split(',')]

        if all(re.match(r'^\w+$', x) for x in names):
            self.cc.scope.update(names)
        else:
            return self.cc.error(ERROR_IDENT)

    each_head_re = r'''(?x)
        ^
        (?P<expr>.+?)
        
        (
            \s+ as \s+
            
            (?P<x> \w+)
            (
                \s*,\s*
                (?P<y> \w+)
            )?
        )?
        
        (
            \s+ index \s+

            (?P<index> \w+)        
            (
                \s*,\s*
                (?P<len> \w+)
            )?
        )?
        $
    '''

    def command_each(self, arg):
        # @each data [as [key,] value] [index i [,len]]

        expr = arg
        vnames = {'x': None, 'y': None, 'index': None, 'len': None}

        m = re.search(self.each_head_re, arg)
        if m:
            vnames = m.groupdict()
            expr = vnames.pop('expr')

        lst = self.cc.new_var()
        e = self.cc.expression.parse(expr)

        self.cc.scope.update(vnames.values())

        if vnames['x'] and vnames['y']:
            init = _f('{} = _RT.iter2({})', lst, e)
            head = _f('for {}, {} in {}:', vnames['x'], vnames['y'], lst)
        elif vnames['x']:
            init = _f('{} = _RT.iter1({})', lst, e)
            head = _f('for {} in {}:', vnames['x'], lst)
        else:
            init = _f('{} = _RT.iter1({})', lst, e)
            head = _f('for {} in {}:', self.cc.new_var(), lst)

        self.cc.code.try_block(
            init,
            _f('{} = []', lst))

        if vnames['index']:
            self.cc.code.add(_f('{} = 0', vnames['index']))
        if vnames['len']:
            self.cc.code.add(_f('{} = len({})', vnames['len'], lst))

        self.cc.code.add(head)
        self.cc.code.begin()

        if vnames['index']:
            self.cc.code.add(_f('{} += 1', vnames['index']))

        cmd, _ = self.cc.parser.parse_until('end', 'else')
        if cmd == 'else':
            self.cc.code.end()
            self.cc.code.add(_f('if not {}:', lst))
            self.cc.code.begin()
            self.cc.parser.parse_until('end')

        self.cc.code.end()

    with_re = r'(?x) (.+?) \b as \s+ (\w+)$'

    def command_with(self, arg):
        # @with expr as var ...flow... @end

        m = re.search(self.with_re, arg)
        if m:
            arg, name = m.groups()
        else:
            name = self.cc.new_var()

        e = self.cc.expression.parse(arg)

        # NB this try-except does not depend on the silent mode

        self.cc.code.raw_try_block(
            _f('{} = {}', name, e),
            _f('{} = None', name)
        )

        self.cc.scope.add(name)

        self.cc.code.add(_f('if not _RT.isempty({}):', name))
        self.cc.code.begin()

        cmd, _ = self.cc.parser.parse_until('end', 'else')

        if cmd == 'else':
            self.cc.code.end()
            self.cc.code.add('else:')
            self.cc.code.begin()
            self.cc.parser.parse_until('end')

        self.cc.code.end()

    def command_include(self, arg):
        # @include path

        path = _relpath(self.cc.parser.current_source.path, arg.strip())
        self.cc.parser.add_source(_read(path, self.cc), path)


class Code:

    def __init__(self, compiler):
        self.cc = compiler
        self.buf = []
        self.textbuf = []
        self.context_vars = set()

    def _emit(self, s):
        if s:
            src = self.cc.parser.current_source
            self.buf.append([src.path, src.lineno, s])

    def _flushbuf(self):
        if self.textbuf:
            self._emit(_f('print({})', repr(''.join(self.textbuf))))
            self.textbuf = []

    def add(self, s):
        self._flushbuf()
        self._emit(s)

    def begin(self):
        self.add('BEGIN')

    def end(self):
        self.add('END')

    def string(self, s):
        self.textbuf.append(s)

    def raw_try_block(self, body, fallback, exc_var=None):
        exc_var = exc_var or self.cc.new_var()
        self.add('try:')
        self.begin()
        for ln in _as_list(body):
            self.add(ln)
        self.end()
        self.add(_f('except Exception as {}:', exc_var))
        self.begin()
        for ln in _as_list(fallback):
            self.add(ln)
        self.end()

    def try_block(self, body, fallback=None):
        if not body:
            return

        if not self.cc.is_silent:
            for ln in _as_list(body):
                self.add(ln)
            return

        exc = self.cc.new_var()

        fallback = _as_list(fallback or [])
        fallback.insert(0, _f('_ERR({})', exc))

        self.raw_try_block(body, fallback, exc)

    def add_context_var(self, v):
        self.context_vars.add(v)

    python_indent = 4

    def python(self):
        self._flushbuf()

        rs = []
        indent = self.python_indent

        def w(lev, s):
            rs.append((' ' * (indent * lev)) + s)

        w(0, _f('def {}(_RT, _CONTEXT, _WARN=None):', self.cc.option('name')))

        if not self.buf:
            w(1, 'return ""')
            return '\n'.join(rs)

        curpath, curline, _ = self.buf[0]

        w(1, _f('_PATH = {}', repr(curpath)))
        w(1, _f('_LINE = {}', repr(curline)))

        w(1, '_BUF = []')
        w(1, 'def _PUSHBUF():')
        w(2, '_BUF.append([])')

        w(1, 'def _POPBUF():')
        w(2, 'b = _BUF.pop()')
        w(2, 'try:')
        w(3, 'return "".join(b)')
        w(2, 'except TypeError:')
        w(3, 'return "".join(str(s) for s in b if s is not None)')

        w(1, 'def _ERRMSG(e):')
        w(2, 'try:')
        w(3, 'name = e.__class__.__name__')
        w(2, 'except:')
        w(3, 'name = repr(e)')
        w(2, 'return "{} in {}:{}".format(name, repr(_PATH), _LINE)')

        w(1, 'def _ERR(e):')
        w(2, 'if _WARN:')
        w(3, '_WARN(_ERRMSG(e))')

        exc = self.cc.new_var()

        if self.cc.is_silent:
            w(1, 'def _GET(obj, prop):')
            w(2, 'try:')
            w(3, 'return _RT.get(obj, prop)')
            w(2, _f('except Exception as {}:', exc))
            w(3, _f('_ERR({})', exc))
            w(3, 'return _RT.undef')
        else:
            w(1, '_GET = _RT.get')

        if self.cc.is_silent:
            w(1, 'def _GETCONTEXT(prop):')
            w(2, 'try:')
            w(3, 'return _CONTEXT[prop] if prop in _CONTEXT else _GLOBALS[prop]')
            w(2, _f('except Exception as {}:', exc))
            w(3, _f('_ERR({})', exc))
            w(3, 'return _RT.undef')
        else:
            w(1, 'def _GETCONTEXT(prop):')
            w(2, 'return _CONTEXT[prop] if prop in _CONTEXT else _GLOBALS[prop]')

        w(1, 'try:')

        vs = self.cc.new_var()

        w(2, _f('{} = {}', vs, sorted(self.context_vars)))
        w(2, _f('_CONTEXT, _GLOBALS = _RT.prepare(_CONTEXT, {})', vs))

        w(2, '_PUSHBUF()')
        w(2, 'print = _BUF[-1].append')

        level = 2

        for path, lineno, op in self.buf:
            if path != curpath:
                w(level, _f('_PATH = {}', repr(path)))
                curpath = path

            if lineno != curline:
                w(level, _f('_LINE = {}', lineno))
                curline = lineno

            if not op:
                continue

            if op == 'BEGIN':
                level += 1
                continue

            if op == 'END':
                level -= 1
                continue

            w(level, op)

            if '_PUSHBUF' in op or '_POPBUF' in op:
                w(level, 'print = _BUF[-1].append')

        exc = self.cc.new_var()

        w(2, 'return _POPBUF()')
        w(1, _f('except Exception as {}:', exc))

        if self.cc.is_silent:
            w(2, _f('_ERR({})', exc))
            w(2, 'return ""')
        else:
            w(2, _f('raise _RT.error_class(_ERRMSG({}))', exc))

        return '\n'.join(rs)


class Compiler:
    def __init__(self, options):
        self.options = dict(_DEFAULT_OPTIONS)
        for k, v in options.items():
            if v is not None:
                self.options[k] = v

    def run(self, text):
        self.parser = Parser(self)
        self.expression = Expression(self)
        self.code = Code(self)
        self.command = Command(self)

        self.globals = set(self.option('globals', []))

        self.scope = set()
        self.blocks = set()
        self.frames = []

        self.num_vars = 0

        self.parser.add_source(text, self.option('path'))
        self.parser.parse_until('eof')

        return self.code.python()

    def error(self, msg, *args):
        _error(msg, self.parser.current_source.path, self.parser.current_source.lineno, *args)

    def option(self, key, default=None):
        return self.options.get(key, default)

    def new_var(self):
        self.num_vars += 1
        return '_' + str(self.num_vars)

    @property
    def is_silent(self):
        return self.option('silent')


##


def _error(msg, path, line, *args):
    code = 'ERROR'
    for k, v in globals().items():
        if k.startswith('ERROR') and v == msg:
            code = k
    if args:
        msg += ' (' + ' '.join(repr(x) for x in args) + ')'
    raise Error(code, msg, path, line)


def _source_location(code, lineno):
    path = '?'
    line = '?'

    for n, ln in enumerate(code.splitlines(), 1):
        if n >= lineno:
            break
        m = re.search(r'_PATH =(.+)', ln)
        if m:
            path = eval(m.group(1))
        m = re.search(r'_LINE =(.+)', ln)
        if m:
            line = eval(m.group(1))

    return path, line


def _findany(pattern, subject, flags=0):
    p = 0
    for m in re.finditer(pattern, subject, flags):
        s, e = m.start(), m.end()
        if s > p:
            yield None, subject[p:s]
        yield m, m.group(0)
        p = e
    if p < len(subject):
        yield None, subject[p:]


def _get_indent(s):
    return len(s) - len(s.lstrip())


def _dedent(s, size):
    if size > 0:
        return re.sub(r'^[ \t]{' + str(size) + '}', '', s)
    return s


def _reindent(buf):
    minlen = 1e20

    for ln in buf:
        if ln.strip():
            minlen = min(minlen, _get_indent(ln))

    return [x[minlen:] for x in buf]


def _relpath(cur_path, path):
    if os.path.isabs(path):
        return path
    d = os.path.dirname(cur_path)
    return os.path.abspath(os.path.join(d, path))


def _read(path, cc=None):
    try:
        with open(path) as fp:
            return fp.read()
    except OSError:
        if cc:
            cc.error(ERROR_FILE, path)
        else:
            raise


def _as_list(s):
    if isinstance(s, (list, tuple)):
        return list(s)
    return [s]


_comma = ', '.join


def _cname(o):
    return o.__class__.__name__


def _f(fmt, *args):
    return fmt.format(*args)

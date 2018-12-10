"""Template compiler"""

import re
import ast
import os
import builtins


class Error(Exception):
    def __init__(self, code, message, path, line):
        msg = '%s: %s in %r on line %s' % (code, message, path, line)
        super().__init__(msg)
        self.code = code
        self.message = msg
        self.path = path
        self.line = line


ERROR_SYNTAX = 'syntax error'
ERROR_BAD_COMMAND = 'unknown command'
ERROR_EOF = 'unexpected end of file'
ERROR_INVALID_DEF = 'function definition error'
ERROR_INVALID_FILTER = 'filter definition error'
ERROR_INVALID_COMMAND = 'command definition error'
ERROR_FILE = 'file not found'
ERROR_USER_COMMAND_SYNTAX = 'invalid user command syntax'
ERROR_LET_SYNTAX = 'invalid let syntax'
ERROR_NOT_SUPPORTED = 'syntax not supported'
ERROR_ARG_NOT_SUPPORTED = 'argument syntax not supported'
ERROR_FILTER = 'invalid or unknown filter'


def compile(text, **opts) -> type:
    cmpl = Compiler(opts)
    code = cmpl.run(text)

    try:
        globs = {cmpl.option('base_name'): cmpl.option('base', object)}
        exec(code, globs)
        return globs[cmpl.option('class_name')]
    except SyntaxError as e:
        loc = _source_location(code, e.lineno - 1)
        _error(ERROR_SYNTAX, loc[0], loc[1], e.msg)


def translate(text, **opts) -> str:
    return Compiler(opts).run(text)


##

class Source:
    def __init__(self, text, path):
        self.lines = text.splitlines(keepends=True) or ['']
        self.maxline = len(self.lines)
        self.lineno = 0
        self.path = path

    def line(self):
        if self.lineno >= self.maxline:
            return None
        self.lineno += 1
        return self.lines[self.lineno - 1]


##

class Compiler:
    DEFAULT_OPTIONS = {
        'base': None,
        'base_name': 'BASE_TEMPLATE',
        'class_name': 'COMPILED_TEMPLATE',
        'comment': '##',
        'defensive': True,
        'extern': [],
        'filter': None,
        'indent': 4,
        'path': '<string>',
        'strip': True,
    }

    def __init__(self, opts=None):
        self.options = dict(self.DEFAULT_OPTIONS)
        self.options.update(opts or {})

    filter_prefix = 'filter_'

    def run(self, text):
        self.code = []
        self.src = [Source(text, self.option('path'))]
        self.textbuf = ''

        self.used_filters = set()
        self.base_filters = set()

        base = self.option('base')
        if base:
            for attr in sorted(vars(base)):
                if attr.startswith(self.filter_prefix):
                    self.base_filters.add(attr[len(self.filter_prefix):])

        self.extern_names = set(self.option('extern', []))
        self.extern_names.add('self')
        self.extern_names.update(vars(builtins))

        self.func_names = set()
        self.local_names = []
        self.frames = []

        self.num_vars = 0

        self.parse_until('eof')
        return self.generate_code()

    ##

    def error(self, msg, *args):
        _error(msg, self.src[-1].path, self.src[-1].lineno, *args)

    def option(self, key, default=None):
        return self.options.get(key, default)

    ##

    def new_var(self):
        self.num_vars += 1
        return '_%d' % self.num_vars

    def _emit(self, s):
        if s:
            self.code.append([self.src[-1].path, self.src[-1].lineno, s])

    def emit(self, s):
        self.flushbuf()
        self._emit(s)

    def flushbuf(self):
        if self.textbuf:
            self._emit('self.put(%r)' % self.textbuf)
            self.textbuf = ''

    def emit_end(self):
        self.emit('END')

    def emit_str(self, s):
        self.textbuf += s

    def emit_check_group(self, gs, fallback=None):
        if not gs:
            return

        if not self.option('defensive'):
            for s in gs:
                self.emit(s)
            return

        v = self.new_var()
        self.emit('try:')
        for s in gs:
            self.emit(s)
        self.emit_end()
        self.emit('except Exception as %s:' % v)
        self.emit('self.err(%d, %s)' % (self.src[-1].lineno, v))
        self.emit(fallback)
        self.emit_end()

    def emit_check(self, s, fallback=None):
        if s:
            self.emit_check_group([s], fallback)

    ##

    def generate_code(self):
        rs = []
        indent = self.option('indent')

        def w(lev, s):
            rs.append((' ' * (indent * lev)) + s)

        w(0, 'class %s(%s):' % (self.option('class_name'), self.option('base_name')))
        w(1, 'defensive = %r' % self.option('defensive'))

        w(1, 'def _render(self, context):')

        self.generate_render_prologue(w, 2)
        self.generate_render_body(w, 2)

        return '\n'.join(rs)

    def generate_render_prologue(self, w, level):
        w(level, '_get = self.get')

        for f in self.used_filters:
            if f in self.base_filters:
                w(level, '%s = self.%s%s' % (f, self.filter_prefix, f))

    def generate_render_body(self, w, level):

        self.flushbuf()
        lastpath = ''
        lastno = -1

        for path, lineno, op in self.code:
            if path != lastpath:
                w(level, 'self.setpath(%r)' % path)
                lastpath = path

            if not op:
                continue

            if op == 'END':
                level -= 1
                continue

            if lineno != lastno:
                w(0, '##%s##%s' % (path, lineno))
                lastno = lineno

            w(level, op)
            if op.endswith(':'):
                level += 1

    ##

    command_re = r'^@(\w+)(.*)'

    def read_line(self):
        while True:
            ln = self.src[-1].line()

            if ln is not None:
                return ln

            if len(self.src) > 1:
                self.src.pop()
                continue

            return None

    def parse_line(self, ln):
        if ln is None:
            return 'eof', ''

        sl = ln.strip()

        cs = self.option('comment')
        if cs and sl.startswith(cs):
            return '', ''

        if sl.startswith('@@'):
            return 'text', sl[1:]

        m = re.search(self.command_re, sl)
        if not m:
            # NB: give the non-stripped line to text()
            return 'text', ln

        cmd = m.group(1).lower()
        # NB: the final : is optional
        arg = m.group(2).lstrip().rstrip(' :')
        return cmd, arg

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

    def ast_expression(self, node):

        def funargs(n):
            return [expr(a) for a in n.args] + [expr(a) for a in n.keywords]

        def use_filter(s):
            if s not in self.base_filters and s not in self.func_names:
                return self.error(ERROR_FILTER, s)
            self.used_filters.add(s)
            return s

        def filter(le, ri):
            t = _cname(ri)

            # xyz | html
            if t == 'Name':
                return '%s(%s)' % (use_filter(ri.id), expr(le))

            # xyz | cut(5)
            if t == 'Call':
                f = ri.func
                if _cname(f) != 'Name':
                    return self.error(ERROR_FILTER, t)
                args = [expr(le)] + funargs(ri)
                return '%s(%s)' % (use_filter(f.id), _comma(args))

            # xyz | '{:1f}'
            if t == 'Str':
                use_filter('format')
                return 'format(%s, %r)' % (expr(le), ri.s)

            return self.error(ERROR_FILTER, t)

        def should_rewrite(n):
            t = _cname(n)
            if t == 'Attribute':
                return should_rewrite(n.value)
            if t == 'Subscript':
                return should_rewrite(n.value)
            if t == 'Name':
                return n.id not in self.extern_names
            return False

        def op(n):
            t = _cname(n)
            if t not in self._operators:
                return self.error(ERROR_NOT_SUPPORTED, t)
            return self._operators[t]

        def getter(a, b):
            return '_get(%s,%r,%d)' % (a, b, self.src[-1].lineno)

        def expr(n):
            t = _cname(n)

            if t == 'NameConstant':
                return repr(n.value)
            if t == 'Num':
                return repr(n.n)
            if t == 'Str':
                return repr(n.s)

            if t == 'List':
                return '[%s]' % _comma(expr(x) for x in n.elts)

            if t == 'Expression':
                return expr(n.body)

            if t == 'Attribute':
                if should_rewrite(n.value):
                    return getter(expr(n.value), n.attr)
                return '%s.%s' % (expr(n.value), n.attr)

            if t == 'Subscript':
                return '%s[%s]' % (expr(n.value), expr(n.slice))
            if t == 'Index':
                return expr(n.value)

            if t == 'Name':
                if should_rewrite(n) and n.id not in self.local_names:
                    return getter('context', n.id)
                return n.id

            if t == 'Call':
                return '%s(%s)' % (expr(n.func), _comma(funargs(n)))
            if t == 'keyword':
                e = expr(n.value)
                if n.arg:
                    return '%s=%s' % (n.arg, e)
                return '**' + e
            if t == 'Starred':
                return '*%s' % expr(n.value)

            if t == 'BinOp':
                if _cname(n.op) == 'BitOr':
                    return filter(n.left, n.right)
                return '%s %s %s' % (expr(n.left), op(n.op), expr(n.right))
            if t == 'BoolOp':
                o = ' %s ' % op(n.op)
                return o.join(expr(x) for x in n.values)
            if t == 'UnaryOp':
                return '%s %s' % (op(n.op), expr(n.operand))

            if t == 'Compare':
                e = expr(n.left)
                for o, c in zip(n.ops, n.comparators):
                    e += ' %s ' % op(o)
                    e += expr(c)
                return e

            return self.error(ERROR_NOT_SUPPORTED, t)

        return expr(node)

    def expression(self, arg, with_default_filter=False):
        try:
            node = ast.parse(arg.strip(), mode='eval')
        except:
            return self.error(ERROR_SYNTAX)

        has_filter = _cname(node.body) == 'BinOp' and _cname(node.body.op) == 'BitOr'

        s = self.ast_expression(node)

        df = self.option('filter')
        if with_default_filter and df and not has_filter:
            return '(%s) | %s' % (s, df)

        return s

    def fix_args(self, args):
        args = args.strip()
        if not args:
            return '()'
        if not args.startswith('('):
            return '(%s)' % args
        return args

    def parse_until(self, *terminators):
        while True:
            cmd, arg = self.parse_line(self.read_line())

            if not cmd:
                continue

            if cmd == 'eof' and 'eof' not in terminators:
                self.error(ERROR_EOF)

            if cmd in terminators:
                return cmd, arg

            if cmd in self.func_names:
                # @func a,b,c => {func(a,b,c)}
                e = self.expression(cmd + self.fix_args(arg))
                self.emit_check('self.put(%s)' % e)
                continue

            base = self.option('base')
            if base:
                fn = getattr(base, 'command_' + cmd, None)
                if fn:
                    # NB: calling an unbound method with the class first arg
                    fn(base, self, arg)
                    continue

            fn = getattr(self, 'command_' + cmd, None)
            if fn:
                fn(arg)
                continue

            self.error(ERROR_BAD_COMMAND, cmd)

    ##

    def command_pragma(self, arg):
        arg = arg.split()
        val = 'true' if len(arg) == 1 else arg[1]

        if val.lower() == 'true':
            val = True
        elif val.lower() == 'false':
            val = False

        self.options[arg[0]] = val

    text_re = r'''(?x) 
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

    def command_text(self, arg):
        s = arg
        if self.option('strip'):
            s = s.strip()
            if arg.endswith('\n'):
                s += '\n'

        for m, val in _findany(self.text_re, s):
            if m:
                e = self.expression(val[1:-1], with_default_filter=True)
                self.emit_check('self.put(%s)' % e)
            else:
                self.emit_str(val)

    def if_head(self, arg):
        v = self.new_var()
        self.emit_check(
            '%s = %s' % (v, self.expression(arg)),
            '%s = None' % v
        )
        self.emit('if %s:' % v)

    def command_if(self, arg):
        """
            since we need 'if' conditions to be statements,
            we have to transform this

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

        self.emit('for %s in "_":' % self.new_var())
        self.if_head(arg)

        while True:
            cmd, arg = self.parse_until('end', 'else', 'elif')
            if cmd == 'elif':
                self.emit('break')
                self.emit_end()
                self.if_head(arg)
                continue
            if cmd == 'end':
                self.emit('break')
                self.emit_end()
                break
            if cmd == 'else':
                self.emit('break')
                self.emit_end()
                self.emit('if True:')
                self.parse_until('end')
                self.emit('break')
                self.emit_end()
                break

        self.emit_end()

    func_re = r'(?x)^ (\w+) \s* (.*) $'

    def command_return(self, arg):
        self.emit('return %s' % self.expression(arg))

    def parse_def_args(self, args):
        # NB: we parse func def as a func call!
        try:
            node = ast.parse('_' + self.fix_args(args), mode='eval')
        except:
            return self.error(ERROR_SYNTAX)

        names = []
        signature = []

        for a in node.body.args:
            t = _cname(a)
            if t == 'Name':
                names.append(a.id)
                signature.append(a.id)
            else:
                return self.error(ERROR_ARG_NOT_SUPPORTED, t)

        for a in node.body.keywords:
            t = _cname(a)
            if t == 'keyword':
                names.append(a.arg)
                signature.append('%s=%s' % (a.arg, self.ast_expression(a.value)))
            else:
                return self.error(ERROR_ARG_NOT_SUPPORTED, t)

        return names, _comma(signature)

    def command_def(self, arg):
        m = re.search(self.func_re, arg)
        if not m:
            self.error(ERROR_INVALID_DEF)

        name, args = m.groups()
        self.func_names.add(name)
        self.local_names.append(name)

        arg_names, signature = self.parse_def_args(args)
        self.frames.append(len(self.local_names))
        self.local_names.extend(arg_names)

        fun = self.new_var()
        res = self.new_var()
        buf = self.new_var()

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

        self.emit('def %s(%s):' % (name, signature))

        self.emit('def %s():' % fun)
        self.parse_until('end')
        self.emit('pass')
        self.emit_end()

        self.emit('self.pushbuf()')
        self.emit('%s = %s()' % (res, fun))
        self.emit('%s = self.popbuf()' % buf)
        self.emit('return %s if %s is None else %s' % (buf, res, res))
        self.emit_end()

        self.local_names = self.local_names[:self.frames.pop()]

    def command_code(self, arg):
        if arg:
            buf = [arg]
        else:
            buf = []
            while True:
                s = self.read_line()
                cmd, _ = self.parse_line(s)
                if cmd == 'end':
                    break
                buf.append(s.rstrip())

        self.emit_check_group(_dedent(buf), 'pass')

    def command_quote(self, arg):
        eof = arg
        while True:
            s = self.read_line()
            cmd, arg = self.parse_line(s)
            if cmd == 'end' and arg == eof:
                break
            self.emit_str(s)


    let_re = r'(?x)^ (\w+) (.*) $'

    def command_let(self, arg):
        m = re.search(self.let_re, arg)
        if not m:
            return self.error(ERROR_LET_SYNTAX)

        name, expr = m.groups()
        expr = expr.strip()
        self.local_names.append(name)

        # short form: @let var expr
        if expr:
            # @let foo = bar, = is optional
            if expr.startswith('='):
                expr = expr[1:]
            self.emit('%s = %s' % (name, self.expression(expr)))
            return

        # long form: @let var ...lines... @end
        self.emit('self.pushbuf()')
        self.parse_until('end')
        self.emit('%s = self.popbuf()' % name)

    loop_head_re = r'''(?x)
        (?P<expr>.+?)
        (
            \s+ as \s+
            
            (?P<x> \w+)
            (
                \s*,\s*
                (?P<y> \w+)
            )?
            \s*
            (
                \(\s*
                    (?P<index> \w+)        
                    (
                        \s*,\s*
                        (?P<len> \w+)
                    )?
                \s*\)
            )?
        )?
        $
    '''

    def command_each(self, arg):
        e = arg
        v = {'x': None, 'y': None, 'index': None, 'len': None}

        m = re.search(self.loop_head_re, arg)
        if m:
            v = m.groupdict()
            e = v.pop('expr')

        lst = self.new_var()
        expr = self.expression(e)

        self.local_names.extend(v.values())

        if v['x'] and v['y']:
            init = '%s = self.iter2(%s)' % (lst, expr)
            head = 'for %s, %s in %s:' % (v['x'], v['y'], lst)
        elif v['x']:
            init = '%s = self.iter1(%s)' % (lst, expr)
            head = 'for %s in %s:' % (v['x'], lst)
        else:
            init = '%s = self.iter1(%s)' % (lst, expr)
            head = 'for %s in %s:' % (self.new_var(), lst)

        self.emit_check(init, '%s = []' % lst)
        if v['index']:
            self.emit('%s = 0' % v['index'])
        if v['len']:
            self.emit('%s = len(%s)' % (v['len'], lst))

        self.emit(head)
        if v['index']:
            self.emit('%s += 1' % v['index'])

        cmd, _ = self.parse_until('end', 'else')
        if cmd == 'else':
            self.emit_end()
            self.emit('if not %s:' % lst)
            self.parse_until('end')
            self.emit_end()

        self.emit_end()

    with_re = r'(?x) (.+?) \b as \s+ (\w+)$'

    def command_with(self, arg):
        name = self.new_var()

        m = re.search(self.with_re, arg)
        if m:
            arg, name = m.groups()

        self.emit_check(
            '%s = %s' % (name, self.expression(arg)),
            '%s = None' % name
        )
        self.emit('if not self.isempty(%s):' % name)

        self.local_names.append(name)

        cmd, _ = self.parse_until('end', 'else')

        if cmd == 'else':
            self.emit_end()
            self.emit('else:')
            self.parse_until('end')

        self.emit_end()

    def command_include(self, arg):
        path = _relpath(self.src[-1].path, arg.strip())
        try:
            with open(path) as fp:
                self.src.append(Source(fp.read(), path))
        except FileNotFoundError:
            self.error(ERROR_FILE, path)


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
    cc = code.splitlines()
    while lineno >= 0:
        if cc[lineno].startswith('##'):
            return cc[lineno].split('##')[1:]
        lineno -= 1
    return '?', '?'


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


def _dedent(buf):
    minlen = 1e20

    for ln in buf:
        sl = ln.lstrip(' ')
        if not sl:
            continue
        minlen = min(minlen, len(ln) - len(sl))

    return [x[minlen:] for x in buf]


def _relpath(cur_path, path):
    if os.path.isabs(path):
        return path
    d = os.path.dirname(cur_path)
    return os.path.abspath(os.path.join(d, path))


_comma = ', '.join


def _cname(o):
    return o.__class__.__name__

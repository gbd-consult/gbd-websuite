import os


# public API

def do(cmd, engine, options, text, path):
    cc = Compiler(engine, options)

    if text is None:
        loc = Location(pos=0, path=__file__, line_num=1)
        text, path = cc.load(None, path or cc.options.path, loc)
    else:
        path = path or cc.options.path or '<string>'

    cc.buf.paste(text, path, cc.buf.pos)

    node = cc.parse()
    if cmd == 'parse':
        return node

    python = cc.translate(node)
    if cmd == 'translate':
        return python

    fn = cc.compile(python)
    if cmd == 'compile':
        return fn


##

class T:
    CONST = 'CONST'
    EOF = 'EOF'
    INVALID = 'INVALID'
    NAME = 'NAME'
    NEWLINE = 'NEWLINE'
    NUMBER = 'NUMBER'
    STRING = 'STRING'


class C:
    NL = '\n'
    WS = ' \r\t\x0b\f'
    WSNL = ' \r\t\x0b\f\n'

    IDENTIFIER_START = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_')
    IDENTIFIER_CONTINUE = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789')

    DIGIT_DEC = set('0123456789')
    DIGIT_HEX = set('0123456789ABCDEFabcdef')
    DIGIT_OCT = set('01234567')
    DIGIT_BIN = set('01')

    QUOTES = set("\'\"")

    STRING_ESCAPES = {
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

    CONST = {
        'True': True,
        'true': True,
        'False': False,
        'false': False,
        'None': None,
        'null': None,
    }

    ADD_OPS = {'+', '-'}
    MUL_OPS = {'*', '/', '//', '%'}
    CMP_OPS = {'==', '!=', '<=', '<', '>=', '>'}
    POWER_OP = {'**'}
    COMPARE_OPS = CMP_OPS | {'in', 'not in'}

    PUNCT = set('=.:,[]{}()|&?!') | POWER_OP | ADD_OPS | MUL_OPS | CMP_OPS

    KEYWORD_OPS = {'and', 'or', 'not', 'in', 'is'}

    KEYWORDS = {
        'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif',
        'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
        'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
    }

    LOOP_KEYWORDS = {'index', 'length'}

    FORMAT_START_SYMBOLS = ':!'

    SAFE_FILTER = {'safe', 'none'}

    ELSE_SYMBOL = 'else'
    ELIF_SYMBOL = 'elif'
    END_SYMBOL = 'end'

    AUX_COMMANDS = {END_SYMBOL, ELSE_SYMBOL, ELIF_SYMBOL}

    DEF_COMMANDS = {'def', 'box', 'mdef', 'mbox'}

    TOK_TYPE = 0
    TOK_POS = 1
    TOK_NEXTPOS = 2
    TOK_SPACE_BEFORE = 3
    TOK_SPACE_AFTER = 4
    TOK_VALUE = 5

    DEFAULT_OPTIONS = dict(
        name='_RENDER_',
        filter=None,
        loader=None,
        strip=False,
        escapes='@@ @ {{ { }} }',
        comment_symbol='@#',
        command_symbol='@',
        inline_open_symbol='{@',
        inline_close_symbol='}',
        inline_start_whitespace=False,
        echo_open_symbol='{',
        echo_close_symbol='}',
        echo_start_whitespace=False,
    )

    PY_INDENT = ' ' * 4
    PY_BEGIN = 1
    PY_END = -1

    PY_MARKER = '##:'

    PY_TEMPLATE = """\
def $name$(_ENGINE, _ARGS, _ERRORHANDLER=None):
    _PATHS = $paths$
    _ENV = _ENGINE.environment(_PATHS, _ARGS, _ERRORHANDLER)
    ARGS = _ENV.ARGS
    print =_ENV.print
    try:
        _ENV.pushbuf()
        $code$
        return _ENV.popbuf()
    except Exception as _0:
        _ENV.error(_0, $loc$)
"""


class Data:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def __getattr__(self, item):
        return None


class Location(Data):
    pos = 0
    path = ''
    path_index = 0
    line_start_pos = 0
    line_num = 0
    column = 0

    def __repr__(self):
        return '[' + repr(self.path) + ',' + repr(self.line_num) + ']'


class Node:
    class And:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_left_binary_op(self)

    class Argument:
        def __init__(self, name, star, expr):
            self.name = name
            self.star = star
            self.expr = expr

        def emit(self, tr: 'Translator'):
            code = tr.emit(self.expr)
            if self.name:
                return self.name + '=' + code
            if self.star == 1:
                return '*' + code
            if self.star == 2:
                return '**' + code
            return code

    class Attr:
        def __init__(self, subject, name):
            self.subject = subject
            self.name = name

        def emit(self, tr: 'Translator'):
            names = []
            node = self

            while isinstance(node, Node.Attr):
                names.append(node.name)
                node = node.subject

            head = tr.emit(node)
            if len(names) == 1:
                return f'_ENV.attr({head}, {names[0]!r})'

            names = list(reversed(names))
            return f'_ENV.attrs({head}, {names!r})'

    class Call:
        def __init__(self, function, args):
            self.function = function
            self.args = args

        def emit(self, tr: 'Translator'):
            args = _comma(tr.emit(a) for a in self.args)
            fn = tr.emit(self.function)
            if not isinstance(self.function, (Node.Name, Node.Attr, Node.Index)):
                fn = _parens(fn)
            return f'{fn}({args})'

    class Comparison:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_comp_binary_op(self)

    class Const:
        def __init__(self, value):
            self.value = value

        def emit(self, tr: 'Translator'):
            return repr(self.value)

    class Dict:
        def __init__(self, items):
            self.items = items

        def emit(self, tr: 'Translator'):
            return '{' + _comma(tr.emit(k) + ':' + tr.emit(v) for k, v in self.items) + '}'

    class IfExpression:
        def __init__(self, cond, yes, no):
            self.cond = cond
            self.yes = yes
            self.no = no

        def emit(self, tr: 'Translator'):
            return (
                    _parens(tr.emit(self.yes))
                    + ' if '
                    + _parens(tr.emit(self.cond))
                    + ' else '
                    + _parens(tr.emit(self.no))
            )

    class Index:
        def __init__(self, subject, index):
            self.subject = subject
            self.index = index

        def emit(self, tr: 'Translator'):
            subj = tr.emit(self.subject)
            index = ':'.join(tr.emit(a) if a else '' for a in self.index)
            return f'{subj}[{index}]'

    class List:
        def __init__(self, items):
            self.items = items

        def emit(self, tr: 'Translator'):
            return '[' + _comma(tr.emit(v) for v in self.items) + ']'

    class Name:
        def __init__(self, ident):
            self.ident = ident

        def emit(self, tr: 'Translator'):
            s = self.ident
            if s in tr.locals:
                return s
            return f'_ENV.get({s!r})'

    class Not:
        def __init__(self, subject, ops):
            self.subject = subject
            self.ops = ops

        def emit(self, tr: 'Translator'):
            return tr.emit_unary_op(self)

    class Number:
        def __init__(self, value):
            self.value = value

        def emit(self, tr: 'Translator'):
            return repr(self.value)

    class Or:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_left_binary_op(self)

    class Param:
        def __init__(self, name, star, expr):
            self.name = name
            self.star = star
            self.expr = expr

        def emit(self, tr: 'Translator'):
            n = self.name
            if self.star == 1:
                return '*' + n
            if self.star == 2:
                return '**' + n
            if self.expr:
                return n + '=' + tr.emit(self.expr)
            return n

    class PipeList:
        def __init__(self, subject, pipes):
            self.subject = subject
            self.pipes = pipes

        def emit(self, tr: 'Translator'):
            code = tr.emit(self.subject)

            for p in self.pipes:
                if isinstance(p, Node.Name):
                    # s | f => f(s)
                    if p.ident in C.SAFE_FILTER:
                        continue
                    fn = tr.emit(p)
                    args = [code]
                elif isinstance(p, Node.Call):
                    # s | f(a, b) => f(s, a, b)
                    fn = tr.emit(p.function)
                    args = [code] + [tr.emit(a) for a in p.args]
                else:
                    # s | whatever => (whatever)(s)
                    fn = _parens(tr.emit(p))
                    args = [code]

                code = f'{fn}({_comma(args)})'

            return code

    class Power:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_right_binary_op(self)

    class Product:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_left_binary_op(self)

    class Sum:
        def __init__(self, subject, pairs):
            self.subject = subject
            self.pairs = pairs

        def emit(self, tr: 'Translator'):
            return tr.emit_left_binary_op(self)

    class Unary:
        def __init__(self, subject, ops):
            self.subject = subject
            self.ops = ops

        def emit(self, tr: 'Translator'):
            return tr.emit_unary_op(self)

    class String:
        def __init__(self, value):
            self.value = value

        def emit(self, tr: 'Translator'):
            return repr(self.value)

    ##

    class Template:
        def __init__(self):
            self.children = []

        def emit(self, tr: 'Translator'):
            return [tr.emit(c) for c in self.children]

    class Location:
        def __init__(self, loc):
            self.loc = loc

        def emit(self, tr: 'Translator'):
            return self

    class Echo:
        def __init__(self, start_pos):
            self.start_pos = start_pos
            self.expr = None
            self.format = None
            self.filter = None

        def parse(self, tp: 'TemplateParser'):
            tp.expr.paren_stack.append('{')
            expr = tp.expr.expect_expression()
            tp.expr.paren_stack.pop()

            tp.buf.skip_ws(with_nl=True)

            fmt = None
            if tp.buf.char() in C.FORMAT_START_SYMBOLS:
                fmt = tp.lex.string_to(tp.cc.options.echo_close_symbol, self.start_pos)

            if not tp.buf.at_string(tp.cc.options.echo_close_symbol):
                raise CompileError('unterminated echo', tp.buf.loc(), tp.buf.loc(self.start_pos))

            tp.buf.next(len(tp.cc.options.echo_close_symbol))

            return self.parse_with_expression(tp, expr, fmt)

        def parse_with_expression(self, tp, expr, fmt):
            self.expr = expr
            self.format = fmt

            has_safe_filter = False
            if isinstance(self.expr, Node.PipeList):
                has_safe_filter = any(isinstance(p, Node.Name) and p.ident in C.SAFE_FILTER for p in self.expr.pipes)
            self.filter = None if has_safe_filter else tp.cc.options.filter

            tp.add_child(self)

        def emit(self, tr: 'Translator'):
            code = tr.emit(self.expr)

            if self.format:
                fn = tr.emit(Node.Name('format'))
                code = f'{fn}({code},{self.format!r})'

            if self.filter:
                fn = tr.emit(Node.Name(self.filter))
                code = f'{fn}({code})'

            return tr.emit_echo(code)

    class Text:
        def __init__(self, text):
            self.text = text

        def emit(self, tr: 'Translator'):
            return self

    class Command:
        def __init__(self, cmd, start_pos):
            self.cmd = cmd
            self.start_pos = start_pos

        def parse(self, tp: 'TemplateParser', is_inline):
            pass

        def parse_elif(self, tp: 'TemplateParser', is_inline):
            raise ValueError()

        def parse_else(self, tp: 'TemplateParser', is_inline):
            raise ValueError()

        def parse_end(self, tp: 'TemplateParser', is_inline):
            raise ValueError()

    class BlockCommand(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.blocks = []
            self.children = []
            self.has_else = False

        def parse_end(self, tp, is_inline):
            self.blocks.append(self.children)
            tp.end_command(self, is_inline)

    class DefineFunction(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.name = None
            self.params = None
            self.expr = None
            self.from_engine = False

        def parse(self, tp, is_inline):
            self.name = tp.expr.expect_name().ident
            self.params = tp.expr.top_level_param_list()

            tp.static_function_bindings[self.name] = [self.cmd, None]

            tok = tp.lex.token(ignore_nl=is_inline)
            if tok[C.TOK_TYPE] == '=':
                self.expr = tp.expr.expect_expression()
            else:
                tp.lex.back(tok)

            tp.expect_end_of_command(is_inline)
            if self.expr:
                tp.add_child(self)
            else:
                tp.begin_command(self)

        def emit(self, tr: 'Translator'):
            tr.locals.add(self.name)
            tr.enter_frame()
            tr.locals.update(p.name for p in self.params)
            params = _comma(tr.emit(p) for p in self.params)

            if self.blocks:
                # 'big' functions are wrapped in push/popbuf to allow explicit returns
                #
                # def userfunc(args):
                #     nul = object()
                #     def fun()
                #         [@return ...]
                #         return nul
                #
                #     _ENV.pushbuf
                #     res = fun()
                #     buf = _ENV.popbuf
                #     return buf if res == nul else res
                #

                nul = tr.var()
                fun = tr.var()
                res = tr.var()
                buf = tr.var()

                code = [
                    f'def {self.name}({params}):',
                    C.PY_BEGIN,
                    f'{nul} = object()',
                    f'def {fun}():',
                    C.PY_BEGIN,
                    [tr.emit(c) for c in self.blocks[0]],
                    f'return {nul}',
                    C.PY_END,
                    f'_ENV.pushbuf()',
                    f'{res} = {fun}()',
                    f'{buf} = _ENV.popbuf()',
                    f'return {buf} if {res} == {nul} else {res}',
                    C.PY_END,
                ]

            else:
                # small function

                res = tr.var()
                code = [
                    f'def {self.name}({params}):',
                    C.PY_BEGIN,
                    tr.emit_assign(res, tr.emit(self.expr)),
                    f'return {res}',
                    C.PY_END,
                ]

            tr.leave_frame()
            return code

    class CallFunctionAsCommand(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.args = None
            self.static_emit_name = ''
            self.block_required = False

        def parse(self, tp, is_inline):
            kind, engine_name = tp.static_function_bindings.get(self.cmd)
            self.static_emit_name = f'_ENGINE.{engine_name}' if engine_name else self.cmd

            if kind == 'box':
                self.block_required = True
                self.args = tp.expr.top_level_arg_list()
                tp.expect_end_of_command(is_inline)
                tp.begin_command(self)
                return

            if kind == 'mbox':
                self.args = tp.expr.top_level_arg_list()
                tp.expect_end_of_command(is_inline)
                text = tp.quoted_content(self.cmd, is_inline)
                self.args.insert(0, Node.String(text))
                tp.add_child(self)
                return

            if kind == 'def':
                self.args = tp.expr.top_level_arg_list()
                tp.expect_end_of_command(is_inline)
                tp.add_child(self)
                return

            if kind == 'mdef':
                tail = tp.command_tail(is_inline)
                self.args = [Node.String(tail)]
                tp.add_child(self)
                return

        def emit(self, tr: 'Translator'):
            fn = self.static_emit_name

            if self.block_required:
                buf = tr.var()
                args = _comma([buf] + [tr.emit(a) for a in self.args])
                return [
                    '_ENV.pushbuf()',
                    [tr.emit(c) for c in self.blocks[0]],
                    f'{buf} = _ENV.popbuf()',
                    tr.emit_echo(f'{fn}({args})')
                ]
            else:
                args = _comma(tr.emit(a) for a in self.args)
                return tr.emit_echo(f'{fn}({args})')

    ##

    class CommandBreakContinue(Command):
        def parse(self, tp, is_inline):
            start_pos = tp.buf.pos
            tp.expect_end_of_command(is_inline)
            if tp.in_loop_context():
                return tp.add_child(self)
            raise CompileError(f"unexpected '{self.cmd}'", tp.buf.loc(start_pos))

        def emit(self, tr: 'Translator'):
            return self.cmd

    class CommandCode(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.lines = []

        def parse(self, tp, is_inline):
            start_pos = tp.buf.pos
            label = tp.command_label(is_inline)
            tp.expect_end_of_command(is_inline)
            text = tp.quoted_content(label, is_inline)
            self.lines = _dedent(text.split(C.NL))

            check = self.lines
            check = ['def _():'] + _indent(check)

            try:
                compile(C.NL.join(check), '', 'exec')
            except SyntaxError as exc:
                loc = tp.buf.loc(start_pos)
                loc.line_num += exc.lineno - 1
                raise CompileError(exc.msg, loc)

            tp.add_child(self)

        def emit(self, tr: 'Translator'):
            nul = tr.var()
            res = tr.var()
            fun = tr.var()
            return [
                f'def {fun}():',
                C.PY_BEGIN,
                self.lines,
                f'return {nul}',
                C.PY_END,
                f'{nul} = object()',
                tr.emit_assign(res, f'{fun}()'),
                f'if {res} != {nul}:',
                C.PY_BEGIN,
                f'return {res}',
                C.PY_END,
            ]

    class CommandDo(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.exprs = []

        def parse(self, tp, is_inline):
            self.exprs = tp.expr.expression_list()
            tp.expect_end_of_command(is_inline)
            tp.add_child(self)

        def emit(self, tr: 'Translator'):
            return tr.emit_try(f'{_comma(tr.emit(a) for a in self.exprs)}')

    class CommandFor(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.subject = None
            self.names = []
            self.extras = {}

        def parse(self, tp, is_inline):
            self.extras = {w: None for w in C.LOOP_KEYWORDS}

            if self.cmd == 'for':
                self.names = [n.ident for n in tp.expr.name_list()]
                tp.expr.expect_token('in')
                self.subject = tp.expr.expect_expression()
                while True:
                    tok = tp.lex.token(ignore_nl=is_inline)
                    if tok[C.TOK_TYPE] == T.NAME and tok[C.TOK_VALUE] in C.LOOP_KEYWORDS:
                        self.extras[tok[C.TOK_VALUE]] = tp.expr.expect_name().ident
                    else:
                        tp.lex.back(tok)
                        break

            if self.cmd == 'each':
                self.subject = tp.expr.expect_expression()
                tp.expr.expect_token('as')
                self.names = [n.ident for n in tp.expr.name_list()]
                for _ in C.LOOP_KEYWORDS:
                    if len(self.names) > 2 and self.names[-2] in C.LOOP_KEYWORDS:
                        self.extras[self.names[-2]] = self.names[-1]
                        self.names = self.names[:-2]

            tp.expect_end_of_command(is_inline)
            tp.begin_command(self)

        def parse_else(self, tp, is_inline):
            if self.has_else:
                raise ValueError()
            self.has_else = True
            self.blocks.append(self.children)
            self.children = []
            tp.expect_end_of_command(is_inline)

        def emit(self, tr: 'Translator'):
            it = tr.var()
            expr = f'_ENV.iter({tr.emit(self.subject)}, {len(self.names)})'
            code = tr.emit_assign(it, expr, fallback='""')

            tr.locals.update(self.names)

            v = self.extras.get('length')
            if v:
                tr.locals.add(v)
                code.append(tr.emit_try(
                    f'{v} = len({it})',
                    f'{v} = 0',
                ))

            cnt = tr.var()
            v = self.extras.get('index')
            if v:
                tr.locals.add(v)
                cnt = v

            code.extend([
                f'{cnt} = 0',
                f'for {_comma(self.names)} in {it}:',
                C.PY_BEGIN,
                f'{cnt} += 1',
                [tr.emit(c) for c in self.blocks[0]],
                C.PY_END
            ])

            if len(self.blocks) > 1:
                code.extend([
                    f'if {cnt} == 0:',
                    C.PY_BEGIN,
                    [tr.emit(c) for c in self.blocks[1]],
                    C.PY_END
                ])

            return code

    class CommandIf(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.conds = []

        def parse(self, tp, is_inline):
            self.conds.append(tp.expr.expect_expression())
            tp.expect_end_of_command(is_inline)
            tp.begin_command(self)

        def parse_elif(self, tp, is_inline):
            self.blocks.append(self.children)
            self.children = []
            self.conds.append(tp.expr.expect_expression())
            tp.expect_end_of_command(is_inline)

        def parse_else(self, tp, is_inline):
            if self.has_else:
                raise ValueError()
            self.has_else = True
            self.blocks.append(self.children)
            self.children = []
            self.conds.append(Node.Const(True))
            tp.expect_end_of_command(is_inline)

        def emit(self, tr: 'Translator'):
            """

                if A
                    aaa
                elif B
                    bbb
                else
                    ccc

            is compiled to

                done = False
                if not done:
                    x = A
                    if x:
                        done = True
                        aaa
                if not done:
                    x = B
                    if x:
                        done = True
                        bbb
                if not done:
                    x = True
                    if x:
                        done = True
                        ccc


            """

            done = tr.var()
            code = [f'{done} = False']
            for cond, block in zip(self.conds, self.blocks):
                cnd = tr.var()
                code.extend([
                    f'if not {done}:',
                    C.PY_BEGIN,
                    tr.emit_assign(cnd, tr.emit(cond)),
                    f'if {cnd}:',
                    C.PY_BEGIN,
                    f'{done} = True',
                    [tr.emit(c) for c in block],
                    C.PY_END,
                    C.PY_END,
                ])

            return code

    class CommandImport(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.names = []

        def parse(self, tp, is_inline):
            self.names = tp.expect_dotted_names()
            tp.expect_end_of_command(is_inline)
            tp.add_child(self)

        def emit(self, tr: 'Translator'):
            tr.locals.add(self.names[0])
            return tr.emit_try('import ' + '.'.join(self.names))

    class CommandInclude(Command):
        def parse(self, tp, is_inline):
            loc = tp.buf.loc()
            tail = _unquote(tp.command_tail(is_inline))
            text, path = tp.cc.load(loc.path, tail, loc)
            tp.buf.paste(text, path, tp.buf.pos)
            tp.reset()

    class CommandLet(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.names = []
            self.exprs = None

        def parse(self, tp, is_inline):
            self.names = [n.ident for n in tp.expr.name_list()]

            tok = tp.lex.token(ignore_nl=is_inline)
            if tok[C.TOK_TYPE] == '=':
                self.exprs = tp.expr.expression_list()
            else:
                tp.lex.back(tok)

            tp.expect_end_of_command(is_inline)
            if self.exprs:
                tp.add_child(self)
            else:
                tp.begin_command(self)

        def emit(self, tr: 'Translator'):
            if self.exprs:
                code = tr.emit_try(
                    f'{_comma(self.names)} = {_comma(tr.emit(a) for a in self.exprs)}',
                    f'{"=".join(self.names)} = None',
                )
            else:
                code = [
                    f'_ENV.pushbuf()',
                    [tr.emit(c) for c in self.blocks[0]],
                    f'{_comma(self.names)} = _ENV.popbuf()',
                ]

            tr.locals.update(self.names)
            return code

    class CommandOption(Command):
        def parse(self, tp, is_inline):
            name = tp.expr.expect_name().ident
            tok = tp.lex.token(ignore_nl=is_inline)
            if tok[C.TOK_TYPE] != '=':
                tp.lex.back(tok)
            pos = tp.buf.pos
            val = tp.expr.expect_expression()
            tp.expect_end_of_command(is_inline)
            if not isinstance(val, (Node.String, Node.Number, Node.Const)):
                raise CompileError(f'invalid option value', tp.buf.loc(pos))
            setattr(tp.cc.options, name, val.value)
            tp.reset()

    class CommandPrint(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.exprs = []

        def parse(self, tp, is_inline):
            args = tp.expr.expression_list()
            tp.expect_end_of_command(is_inline)

            start = True
            for arg in args:
                if not start:
                    tp.add_raw_text(' ')
                start = False
                Node.Echo(self.start_pos).parse_with_expression(tp, arg, None)
            if not is_inline:
                tp.add_raw_text('\n')

    class CommandComment(Command):
        def parse(self, tp, is_inline):
            label = tp.command_label(is_inline)
            tp.expect_end_of_command(is_inline)
            text = tp.quoted_content(label, is_inline)

    class CommandQuote(Command):
        def parse(self, tp, is_inline):
            label = tp.command_label(is_inline)
            tp.expect_end_of_command(is_inline)
            text = tp.quoted_content(label, is_inline)
            tp.add_raw_text(text)

    class CommandReturn(Command):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.expr = None

        def parse(self, tp, is_inline):
            self.expr = tp.expr.expression()
            tp.expect_end_of_command(is_inline)
            return tp.add_child(self)

        def emit(self, tr: 'Translator'):
            code = 'return'
            if self.expr:
                code += ' ' + tr.emit(self.expr)
            return code

    class CommandWith(BlockCommand):
        def __init__(self, cmd, start_pos):
            super().__init__(cmd, start_pos)
            self.subject = None
            self.alias = None
            self.inverted = False

        def parse(self, tp, is_inline):
            self.inverted = self.cmd == 'without'
            self.subject = tp.expr.expect_expression()

            tok = tp.lex.token(ignore_nl=is_inline)
            if tok[C.TOK_TYPE] == 'as':
                self.alias = tp.expr.expect_name().ident
            else:
                tp.lex.back(tok)

            tp.expect_end_of_command(is_inline)
            tp.begin_command(self)

        def parse_else(self, tp, is_inline):
            if self.has_else:
                raise ValueError()
            self.has_else = True
            self.blocks.append(self.children)
            self.children = []
            tp.expect_end_of_command(is_inline)

        def emit(self, tr: 'Translator'):
            if self.alias:
                tr.locals.add(self.alias)
                var = self.alias
            else:
                var = tr.var()

            op = "" if self.inverted else "not"

            code = [
                tr.emit_assign(var, tr.emit(self.subject), mute=True),
                f'if {op} _ENV.isempty({var}):',
                C.PY_BEGIN,
                [tr.emit(c) for c in self.blocks[0]],
                C.PY_END
            ]

            if len(self.blocks) > 1:
                code.extend([
                    'else:',
                    C.PY_BEGIN,
                    [tr.emit(c) for c in self.blocks[1]],
                    C.PY_END
                ])

            return code

    COMMANDS = {
        'def': DefineFunction,
        'box': DefineFunction,
        'mdef': DefineFunction,
        'mbox': DefineFunction,

        'break': CommandBreakContinue,
        'code': CommandCode,
        'comment': CommandComment,
        'continue': CommandBreakContinue,
        'do': CommandDo,
        'each': CommandFor,
        'for': CommandFor,
        'if': CommandIf,
        'import': CommandImport,
        'include': CommandInclude,
        'let': CommandLet,
        'option': CommandOption,
        'print': CommandPrint,
        'quote': CommandQuote,
        'return': CommandReturn,
        'skip': CommandComment,
        'with': CommandWith,
        'without': CommandWith,
    }


class CompileError(ValueError):
    def __init__(self, message, loc: Location, start_loc: Location = None):
        self.path = loc.path if loc else '<unknown>'
        self.line = loc.line_num if loc else '?'
        if loc:
            message += ' in ' + _path_line(loc.path, loc.line_num)
        if start_loc:
            message += ' (started in ' + _path_line(start_loc.path, start_loc.line_num) + ')'
        super().__init__(message)
        self.message = message


class Buffer:
    def __init__(self):
        self.length = 0
        self.line_pointers = []
        self.location_cache = {}
        self.paths = []
        self.pos = 0
        self.text = ''

    def paste(self, text, path, pos):
        try:
            path_index = self.paths.index(path)
        except ValueError:
            self.paths.append(path)
            path_index = len(self.paths) - 1

        if not text:
            if not self.line_pointers:
                self.line_pointers = [[0, path_index, 1]]
            return

        text_len = len(text)
        lps = self.line_pointers

        for lp in lps:
            if lp[0] >= pos:
                lp[0] += text_len

        line_num = 1
        p = 0
        while p < text_len:
            lps.append([pos + p, path_index, line_num])
            p = text.find(C.NL, p)
            if p < 0:
                break
            p += 1
            line_num += 1

        lps.sort()
        self.text = self.text[:pos] + text + self.text[pos:]
        self.length = len(self.text)
        self.location_cache = {}

    def loc(self, pos=None):
        if pos is None:
            pos = self.pos

        if pos in self.location_cache:
            return self.location_cache[pos]

        lptr = self.line_pointers[self.lptr_index(pos)]
        line_start_pos, path_index, line_num = lptr
        loc = Location(
            line_start_pos=line_start_pos,
            path_index=path_index,
            path=self.paths[path_index],
            line_num=line_num,
            column=pos - line_start_pos,
            pos=pos)

        self.location_cache[pos] = loc
        return loc

    def lptr_index(self, pos):
        lps = self.line_pointers
        le = 0
        ri = len(lps) - 1
        while le <= ri:
            m = (le + ri) >> 1
            if lps[m][0] < pos:
                le = m + 1
            elif lps[m][0] > pos:
                ri = m - 1
            else:
                return m
        return le - 1

    def eof(self):
        return self.pos >= self.length

    def at_line_start(self, pos=None):
        if pos is None:
            pos = self.pos
        lptr = self.line_pointers[self.lptr_index(pos)]
        p = lptr[0]
        while p < self.pos:
            if self.text[p] not in C.WS:
                return False
            p += 1
        return True

    def at_space(self):
        return self.char() in C.WSNL

    def at_string(self, s):
        if not s:
            return False
        return self.text.startswith(s, self.pos)

    def char(self):
        try:
            return self.text[self.pos]
        except IndexError:
            return ''

    def next(self, n=1):
        self.pos += n

    def to(self, p):
        self.pos = p

    def line_tail(self):
        li = self.lptr_index(self.pos)
        if li < len(self.line_pointers) - 1:
            p = self.line_pointers[li + 1][0] - 1
            s = self.text[self.pos:p]
            self.pos = p + 1
        else:
            s = self.text[self.pos:]
            self.pos = self.length
        return s.strip()

    def find(self, s, pos):
        return self.text.find(s, pos)

    def skip_ws(self, with_nl: bool):
        p = self.pos
        chars = C.WSNL if with_nl else C.WS
        while p < self.length:
            if self.text[p] not in chars:
                break
            p += 1
        if p == self.pos:
            return False
        self.pos = p
        return True

    def strpbrk(self, chars):
        p = self.pos
        while p < self.length:
            if self.text[p] in chars:
                break
            p += 1
        return p

    def slice(self, a, b):
        return self.text[a:b]


class Lexer:
    def __init__(self, buf: Buffer):
        self.buf = buf
        self.token_cache = {}

    def token(self, ignore_nl: bool):
        pos = self.buf.pos
        if pos not in self.token_cache:
            self.token_cache[pos] = self.token2(ignore_nl)
        self.buf.to(self.token_cache[pos][C.TOK_NEXTPOS])
        return self.token_cache[pos]

    def back(self, tok):
        self.buf.to(tok[C.TOK_POS])

    def token2(self, ignore_nl):
        value = None
        pos = self.buf.pos

        space_before = self.buf.skip_ws(with_nl=ignore_nl)

        if self.buf.eof():
            typ = T.EOF
        elif self.buf.char() == C.NL:
            typ = T.NEWLINE
        else:
            tv = self.token3()
            if tv:
                typ = tv[0]
                value = tv[1]
            else:
                self.buf.to(pos)
                typ = T.INVALID

        return [
            typ,
            pos,
            self.buf.pos,
            space_before,
            self.buf.at_space(),
            value
        ]

    def token3(self):
        ch = self.buf.char()

        if ch in C.IDENTIFIER_START:
            s = self.identifier()
            if s in C.KEYWORDS:
                if s == 'not':
                    pos2 = self.buf.pos
                    self.buf.skip_ws(with_nl=False)
                    s2 = self.identifier()
                    if s2 == 'in':
                        return 'not in', None
                    self.buf.to(pos2)
                return s, None
            if s in C.CONST:
                return T.CONST, C.CONST[s]
            return T.NAME, s

        if ch in C.QUOTES:
            return T.STRING, self.string()

        if ch in C.DIGIT_DEC:
            return T.NUMBER, self.number()

        if ch in C.PUNCT:
            return self.punctuation(), None

    def identifier(self):
        chars = ''

        while not self.buf.eof():
            ch = self.buf.char()
            if ch not in C.IDENTIFIER_CONTINUE:
                break
            chars += ch
            self.buf.next()

        return chars

    def punctuation(self):
        c1 = self.buf.char()
        if c1 in C.PUNCT:
            self.buf.next()
            c2 = self.buf.char()
            if c1 + c2 in C.PUNCT:
                self.buf.next()
                return c1 + c2
            return c1

    def string(self):
        quot = self.buf.char()
        pos = self.buf.pos
        self.buf.next()
        chars = self.string_to(quot, pos)
        self.buf.next()
        return chars

    def string_to(self, end_sym, start_pos):
        chars = ''

        while True:
            if self.buf.at_string(end_sym):
                break

            ch = self.buf.char()
            if not ch or ch == C.NL:
                raise CompileError('unterminated string', self.buf.loc(self.buf.pos), self.buf.loc(start_pos))
            if ch == '\\':
                self.buf.next()
                chars += self.string_escape()
            else:
                self.buf.next()
                chars += ch

        return chars

    def string_escape(self):
        ch = self.buf.char()

        if ch in C.STRING_ESCAPES:
            self.buf.next()
            return C.STRING_ESCAPES[ch]

        if ch == 'x' or ch == 'X' or ch == 'u' or ch == 'U':
            self.buf.next()
            code = self.unicode_escape(ch)
            if 0 <= code < 0x110000:
                return chr(code)
            raise CompileError('invalid escape sequence', self.buf.loc(self.buf.pos))

        self.buf.next()
        return ch

    def unicode_escape(self, prefix):
        # \xXX
        if prefix == 'x' or prefix == 'X':
            return self.hexcode(2, 2)

        # \UXXXXXXXX
        if prefix == 'U':
            return self.hexcode(8, 8)

        # \u{XX...}
        if prefix == 'u' and self.buf.char() == '{':
            self.buf.next()
            code = self.hexcode(1, 8)
            if self.buf.char() == '}':
                self.buf.next()
                return code
            return -1

        # \uXXXX
        if prefix == 'u':
            return self.hexcode(4, 4)

        return -1

    def hexcode(self, minlen, maxlen):
        chars = ''

        while len(chars) < maxlen:
            ch = self.buf.char()
            if ch in C.DIGIT_HEX:
                self.buf.next()
                chars += ch
            else:
                break

        if minlen <= len(chars) <= maxlen:
            return int(chars, 16)

        return -1

    def number(self):
        ch = self.buf.char()

        if ch == '0':
            pos = self.buf.pos
            self.buf.next()
            ch = self.buf.char()
            if ch == 'x' or ch == 'X':
                self.buf.next()
                return self.nondec_number(C.DIGIT_HEX, 16)
            if ch == 'o' or ch == 'O':
                self.buf.next()
                return self.nondec_number(C.DIGIT_OCT, 8)
            if ch == 'b' or ch == 'B':
                self.buf.next()
                return self.nondec_number(C.DIGIT_BIN, 2)
            self.buf.to(pos)

        return self.dec_number()

    def dec_number(self):
        i = self.digit_seq(C.DIGIT_DEC)

        f = ''
        if self.buf.char() == '.':
            self.buf.next()
            f = self.digit_seq(C.DIGIT_DEC)
            if not f:
                raise self.number_error()

        if not i and not f:
            raise self.number_error()

        e = esign = ''
        ch = self.buf.char()
        if ch == 'e' or ch == 'E':
            self.buf.next()
            ch = self.buf.char()
            if ch == '+' or ch == '-':
                esign = ch
                self.buf.next()
            e = self.digit_seq(C.DIGIT_DEC)
            if not e:
                raise self.number_error()

        if f or e:
            return float((i or '0') + '.' + (f or '0') + 'E' + (esign or '') + (e or '0'))

        return int(i, 10)

    def nondec_number(self, digits, base):
        n = self.digit_seq(digits)
        if not n:
            raise self.number_error()
        return int(n, base)

    def digit_seq(self, digits):
        chars = ''

        while True:
            ch = self.buf.char()
            if ch in digits:
                chars += ch
                self.buf.next()
            elif ch == '_':
                self.buf.next()
            else:
                break

        return chars

    def number_error(self):
        return CompileError('invalid numeric literal', self.buf.loc())


class ExpressionParser:
    def __init__(self, lex: Lexer):
        self.lex = lex
        self.paren_stack = []

    def expression(self):
        subject = self.ifexpr()
        if not subject:
            return

        pipes = []

        while True:
            tok = self.token()
            if tok[C.TOK_TYPE] == '|' and tok[C.TOK_SPACE_BEFORE] == tok[C.TOK_SPACE_AFTER]:
                pipes.append(self.expect(self.primary))
            else:
                self.lex.back(tok)
                break

        if not pipes:
            return subject

        return Node.PipeList(subject, pipes)

    def ifexpr(self):
        yes = self.orexpr()
        if not yes:
            return

        tok = self.token()
        if tok[C.TOK_TYPE] == 'if':
            cond = self.expect(self.orexpr)
            self.expect_token('else')
            return Node.IfExpression(cond, yes, self.expect(self.ifexpr))
        self.lex.back(tok)

        return yes

    def orexpr(self):
        return self.binary_op(Node.Or, {'or'}, self.andexpr)

    def andexpr(self):
        return self.binary_op(Node.And, {'and'}, self.notexpr)

    def notexpr(self):
        return self.unary_op(Node.Not, {'not'}, self.comparison)

    def comparison(self):
        return self.binary_op(Node.Comparison, C.COMPARE_OPS, self.sum)

    def sum(self):
        return self.binary_op(Node.Sum, C.ADD_OPS, self.product)

    def product(self):
        return self.binary_op(Node.Product, C.MUL_OPS, self.power)

    def power(self):
        return self.binary_op(Node.Power, C.POWER_OP, self.unary)

    def unary(self):
        return self.unary_op(Node.Unary, C.ADD_OPS, self.primary)

    def primary(self):
        node = self.atom()
        if not node:
            return

        while True:
            tok = self.token()

            if tok[C.TOK_TYPE] == '.' and not tok[C.TOK_SPACE_BEFORE]:
                node = Node.Attr(node, self.expect_name().ident)
                continue

            if tok[C.TOK_TYPE] == '(' and not tok[C.TOK_SPACE_BEFORE]:
                self.push_paren('(')
                node = Node.Call(node, self.generic_list(self.arg_list_item))
                self.pop_paren()
                continue

            if tok[C.TOK_TYPE] == '[' and not tok[C.TOK_SPACE_BEFORE]:
                self.push_paren('[')
                node = Node.Index(node, self.index_list())
                self.pop_paren()
                continue

            self.lex.back(tok)
            break

        return node

    def atom(self):
        tok = self.token()

        if tok[C.TOK_TYPE] == T.STRING:
            return Node.String(tok[C.TOK_VALUE])

        if tok[C.TOK_TYPE] == T.NUMBER:
            return Node.Number(tok[C.TOK_VALUE])

        if tok[C.TOK_TYPE] == T.CONST:
            return Node.Const(tok[C.TOK_VALUE])

        if tok[C.TOK_TYPE] == T.NAME:
            return Node.Name(tok[C.TOK_VALUE])

        if tok[C.TOK_TYPE] == '(':
            self.push_paren('(')
            e = self.expression()
            self.pop_paren()
            return e

        if tok[C.TOK_TYPE] == '[':
            self.push_paren('[')
            items = self.generic_list(self.expression)
            self.pop_paren()
            return Node.List(items)

        if tok[C.TOK_TYPE] == '{':
            self.push_paren('{')
            items = self.generic_list(self.dict_item)
            self.pop_paren()
            return Node.Dict(items)

        self.lex.back(tok)

    # lists

    def generic_list_with_opt_parens(self, item_fn):
        tok = self.token()

        if tok[C.TOK_TYPE] == '(':
            self.push_paren('(')
            res = self.generic_list(item_fn)
            self.pop_paren()
            return res

        self.lex.back(tok)
        if self.lex.buf.at_space():
            return self.generic_list(item_fn)

        return []

    def generic_list(self, fn):
        items = []
        nl = bool(self.paren_stack)

        while True:
            a = fn()
            if not a:
                break
            items.append(a)

            pos = self.lex.buf.pos
            has_delim = self.lex.buf.skip_ws(with_nl=nl)
            if self.lex.buf.char() == ',':
                has_delim = True
                self.lex.buf.next()
                self.lex.buf.skip_ws(with_nl=nl)
            if not has_delim:
                self.lex.buf.to(pos)
                break

        return items

    def top_level_arg_list(self):
        return self.generic_list_with_opt_parens(self.arg_list_item)

    def arg_list_item(self):
        tok = self.token()

        if tok[C.TOK_TYPE] == '*' and not tok[C.TOK_SPACE_AFTER]:
            return Node.Argument(None, 1, self.expect_expression())

        if tok[C.TOK_TYPE] == '**' and not tok[C.TOK_SPACE_AFTER]:
            return Node.Argument(None, 2, self.expect_expression())

        if tok[C.TOK_TYPE] == T.NAME:
            if self.arg_equal():
                return Node.Argument(tok[C.TOK_VALUE], 0, self.expect_expression())

        self.lex.back(tok)

        expr = self.expression()
        if expr:
            return Node.Argument(None, 0, expr)

    def top_level_param_list(self):
        return self.generic_list_with_opt_parens(self.param_list_item)

    def param_list_item(self):
        tok = self.token()

        if tok[C.TOK_TYPE] == T.NAME:
            if self.arg_equal():
                return Node.Param(tok[C.TOK_VALUE], 0, self.expect_expression())
            return Node.Param(tok[C.TOK_VALUE], 0, None)

        if tok[C.TOK_TYPE] == '*' and not tok[C.TOK_SPACE_AFTER]:
            n = self.expect_name()
            return Node.Param(n.ident, 1, None)

        if tok[C.TOK_TYPE] == '**' and not tok[C.TOK_SPACE_AFTER]:
            n = self.expect_name()
            return Node.Param(n.ident, 2, None)

        self.lex.back(tok)

    def name_list(self):
        ls = self.generic_list(self.name_list_item)
        if not ls:
            raise CompileError('name expected', self.lex.buf.loc())
        return ls

    def name_list_item(self):
        tok = self.token()
        if tok[C.TOK_TYPE] == T.NAME:
            return Node.Name(tok[C.TOK_VALUE])
        self.lex.back(tok)

    def expression_list(self):
        ls = self.generic_list(self.expression_list_item)
        if not ls:
            raise CompileError('expression expected', self.lex.buf.loc())
        return ls

    def expression_list_item(self):
        pos = self.lex.buf.pos
        e = self.expression()
        if e:
            return e
        self.lex.buf.to(pos)

    def index_list(self):
        args = [self.expression()]
        if self.slice_colon():
            args.append(self.expression())
            if self.slice_colon():
                args.append(self.expression())

        return args

    def arg_equal(self):
        tok = self.token()
        if tok[C.TOK_TYPE] == '=' and not tok[C.TOK_SPACE_BEFORE] and not tok[C.TOK_SPACE_AFTER]:
            return True
        self.lex.back(tok)
        return False

    def slice_colon(self):
        tok = self.token()
        if tok[C.TOK_TYPE] == ':':
            return True
        self.lex.back(tok)
        return False

    def dict_item(self):
        k = self.expression()
        if k:
            self.expect_token(':')
            return [k, self.expect_expression()]

    # expression utils

    def unary_op(self, typ, operators, arg_fn):
        pos = self.lex.buf.pos
        ops = []

        while True:
            tok = self.token()
            if tok[C.TOK_TYPE] in operators and (tok[C.TOK_TYPE] in C.KEYWORD_OPS or not tok[C.TOK_SPACE_AFTER]):
                ops.append(tok[C.TOK_TYPE])
            else:
                self.lex.back(tok)
                break

        subject = arg_fn()
        if not subject:
            self.lex.buf.to(pos)
            return

        if not ops:
            return subject

        return typ(subject, ops)

    def binary_op(self, typ, operators, arg_fn):
        subject = arg_fn()
        if not subject:
            return

        pairs = []

        while True:
            tok = self.token()
            if tok[C.TOK_TYPE] in operators and tok[C.TOK_SPACE_BEFORE] == tok[C.TOK_SPACE_AFTER]:
                pairs.append([tok[C.TOK_TYPE], self.expect(arg_fn)])
            else:
                self.lex.back(tok)
                break

        if not pairs:
            return subject

        return typ(subject, pairs)

    def expect(self, fn):
        node = fn()
        if node:
            return node
        raise CompileError(f'{fn.__name__!r} expected', self.lex.buf.loc())

    def expect_token(self, typ):
        tok = self.token()
        if tok[C.TOK_TYPE] == typ:
            return tok
        raise CompileError(f'expected {typ!r}, found {tok[C.TOK_TYPE]!r}', self.lex.buf.loc())

    def expect_expression(self):
        return self.expect(self.expression)

    def expect_name(self):
        tok = self.expect_token(T.NAME)
        return Node.Name(tok[C.TOK_VALUE])

    def push_paren(self, s):
        self.paren_stack.append(s)

    def pop_paren(self):
        if not self.paren_stack:
            raise CompileError('unbalanced parentheses', self.lex.buf.loc())
        p = self.paren_stack.pop()
        if p == '(':
            self.expect_token(')')
        if p == '[':
            self.expect_token(']')
        if p == '{':
            self.expect_token('}')

    def token(self):
        return self.lex.token(bool(self.paren_stack))


class TemplateParser:
    def __init__(self, compiler: 'Compiler'):
        self.cc = compiler
        self.buf = self.cc.buf
        self.lex = Lexer(self.buf)
        self.expr = ExpressionParser(self.lex)
        self.stack = []
        self.special_chars = ''
        self.elements = []
        self.escape_dct = {}

        self.static_function_bindings = {}
        for a in dir(self.cc.engine):
            if '_' in a:
                cmd, _, name = a.partition('_')
                if cmd in C.DEF_COMMANDS:
                    self.static_function_bindings[name] = [cmd, a]

        self.reset()

    def reset(self):
        self.buf.location_cache = {}
        self.lex.token_cache = {}

        self.escape_dct = dict(zip(
            self.cc.options.escapes.split()[::2],
            self.cc.options.escapes.split()[1::2]
        ))

        self.special_chars = ''.join(set(
            self.cc.options.comment_symbol
            + self.cc.options.command_symbol
            + self.cc.options.inline_open_symbol
            + self.cc.options.inline_close_symbol
            + self.cc.options.echo_open_symbol
            + self.cc.options.echo_close_symbol
            + ''.join(self.cc.options.escapes.split())
        ))

        # make sure longest parsers come first
        by_len = [
            [self.cc.options.inline_open_symbol, self.inline_element],
            [self.cc.options.echo_open_symbol, self.echo_element],
            [self.cc.options.comment_symbol, self.comment_element],
            [self.cc.options.command_symbol, self.command_element],
        ]
        by_len.sort(key=lambda h: len(h[0]), reverse=True)

        self.elements = [self.text_element, self.escape_element] + [h[1] for h in by_len]

    def parse(self):
        self.stack = [Node.Template()]
        self.add_child(Node.Location(self.buf.loc(0)))

        while not self.buf.eof():
            if not any(f() for f in self.elements):
                self.add_raw_text(self.buf.char())
                self.buf.next()

        if len(self.stack) > 1:
            raise CompileError(
                f'missing {C.END_SYMBOL!r}',
                self.buf.loc(),
                self.buf.loc(self.top().start_pos))

        return self.stack[0]

    def text_element(self):
        p = self.buf.strpbrk(self.special_chars)
        if p > self.buf.pos:
            self.add_text(self.buf.slice(self.buf.pos, p), self.buf.pos)
            self.buf.to(p)
            return True

    def escape_element(self):
        for esc, repl in self.escape_dct.items():
            if self.buf.at_string(esc):
                self.add_raw_text(repl)
                self.buf.next(len(esc))
                return True

    def echo_element(self):
        if not self.buf.at_string(self.cc.options.echo_open_symbol):
            return False
        return self.inline_or_echo_element(
            self.cc.options.echo_open_symbol,
            self.cc.options.echo_close_symbol,
            self.cc.options.echo_start_whitespace,
            self.parse_echo)

    def inline_element(self):
        if not self.buf.at_string(self.cc.options.inline_open_symbol):
            return False
        return self.inline_or_echo_element(
            self.cc.options.inline_open_symbol,
            self.cc.options.inline_close_symbol,
            self.cc.options.inline_start_whitespace,
            self.parse_inline_command)

    def inline_or_echo_element(self, open_sym, close_sym, whitespace_flag, parse_func):
        start_pos = self.buf.pos

        self.buf.next(len(open_sym))
        has_ws = self.buf.skip_ws(with_nl=True)

        # { at EOF or followed by whitespace
        if self.buf.eof() or (has_ws and not whitespace_flag):
            self.add_text(self.buf.slice(start_pos, self.buf.pos), start_pos)
            return True

        # { whitespace... }
        if self.buf.at_string(close_sym):
            self.buf.next(len(close_sym))
            self.add_text(self.buf.slice(start_pos, self.buf.pos), start_pos)
            return True

        return parse_func(start_pos)

    def command_element(self):
        if not (self.buf.at_line_start() and self.buf.at_string(self.cc.options.command_symbol)):
            return False
        start_pos = self.buf.pos
        self.buf.next(len(self.cc.options.command_symbol))
        self.parse_command(start_pos, is_inline=False)
        return True

    def comment_element(self):
        if not (self.buf.at_line_start() and self.buf.at_string(self.cc.options.comment_symbol)):
            return False
        self.strip_last_indent()
        _ = self.buf.line_tail()
        return True

    def parse_echo(self, start_pos):
        self.add_child(Node.Location(self.buf.loc(start_pos)))
        Node.Echo(start_pos).parse(self)
        return True

    def parse_inline_command(self, start_pos):
        return self.parse_command(start_pos, is_inline=True)

    def parse_command(self, start_pos, is_inline):
        cmd = self.lex.identifier()
        if not cmd:
            self.add_text(self.buf.slice(start_pos, self.buf.pos), start_pos)
            return True

        if not is_inline:
            self.strip_last_indent()

        self.add_child(Node.Location(self.buf.loc(start_pos)))

        # custom command?
        p = self.static_function_bindings.get(cmd)
        if p:
            Node.CallFunctionAsCommand(cmd, start_pos).parse(self, is_inline)
            return True

        # else, elif, end?
        if cmd in C.AUX_COMMANDS:
            fn = getattr(self.top(), 'parse_' + cmd, None)
            try:
                fn(self, is_inline)
                return True
            except (TypeError, ValueError):
                raise CompileError(f'unexpected {cmd!r}', self.buf.loc(start_pos))

        # built-in command?
        cls = Node.COMMANDS.get(cmd)
        if cls:
            cls(cmd, start_pos).parse(self, is_inline)
            return True

        raise CompileError(f'unknown command {cmd!r}', self.buf.loc(start_pos))

    def add_text(self, text, pos):
        if self.cc.options.strip:
            lines = text.split(C.NL)
            n = 0 if self.buf.at_line_start(pos) else 1
            while n < len(lines):
                lines[n] = lines[n].lstrip()
                n += 1
            text = C.NL.join(lines)
        return self.add_child(Node.Text(text))

    def add_raw_text(self, text):
        return self.add_child(Node.Text(text))

    def strip_last_indent(self):
        last = self.top().children[-1] if self.top().children else None
        if isinstance(last, Node.Text):
            last.text = last.text.rstrip(C.WS)
            if not last.text:
                self.top().children.pop()

    def command_tail(self, is_inline):
        if is_inline:
            p = self.buf.find(self.cc.options.inline_close_symbol, self.buf.pos)
            if p < 0:
                raise CompileError('unterminated command', self.buf.loc())
            s = self.buf.slice(self.buf.pos, p)
            self.buf.to(p + len(self.cc.options.inline_close_symbol))
        else:
            p = self.buf.find(C.NL, self.buf.pos)
            if p < 0:
                s = self.buf.slice(self.buf.pos, self.buf.length)
                self.buf.to(self.buf.length)
            else:
                s = self.buf.slice(self.buf.pos, p)
                self.buf.to(p + 1)

        return s.strip()

    def expect_end_of_command(self, is_inline):
        tail = self.command_tail(is_inline)
        if tail:
            tail = _cut(tail, 10)
            raise CompileError(f'expected end of command, found {tail!r}', self.buf.loc())

    def expect_dotted_names(self):
        start_pos = self.buf.pos
        names = []
        expr = self.expr.expect_expression()
        while True:
            if isinstance(expr, Node.Attr):
                names.insert(0, expr.name)
                expr = expr.subject
            elif isinstance(expr, Node.Name):
                names.insert(0, expr.ident)
                break
            else:
                raise CompileError('qualified name expected', self.buf.loc(start_pos))
        return names

    def command_label(self, is_inline):
        s = ''
        if self.buf.at_space():
            self.buf.skip_ws(with_nl=is_inline)
            s = self.lex.identifier()
        return s

    def quoted_content(self, label, is_inline):
        start_pos = self.buf.pos

        end_sym = (self.cc.options.inline_open_symbol if is_inline else self.cc.options.command_symbol) + C.END_SYMBOL

        while not self.buf.eof():
            end_pos = self.buf.find(end_sym, self.buf.pos)
            if end_pos < 0:
                break
            at_line_start = self.buf.at_line_start()
            self.buf.to(end_pos + len(end_sym))
            if not is_inline and not at_line_start:
                continue
            end_label = self.command_label(is_inline)
            tail = self.command_tail(is_inline)
            if end_label == label and not tail:
                s = self.buf.slice(start_pos, end_pos)
                if not is_inline:
                    s = s.rstrip(C.WS)
                return s

        raise CompileError(f'missing {C.END_SYMBOL!r}', self.buf.loc(), self.buf.loc(start_pos))

    def top(self):
        return self.stack[-1]

    def add_child(self, node):
        self.top().children.append(node)
        return node

    def begin_command(self, node):
        node = self.add_child(node)
        self.stack.append(node)
        return node

    def end_command(self, node, is_inline):
        label = self.command_label(is_inline)
        if label and label != node.cmd:
            a = C.END_SYMBOL + ' ' + node.cmd
            b = C.END_SYMBOL + ' ' + label
            raise CompileError(f'expected {a!r}, found {b!r}', self.buf.loc())
        self.expect_end_of_command(is_inline)
        self.stack.pop()

    def in_loop_context(self):
        for node in reversed(self.stack):
            if isinstance(node, Node.CommandFor):
                return True
            if isinstance(node, Node.DefineFunction):
                break
        return False

    def in_def_context(self):
        for node in reversed(self.stack):
            if isinstance(node, Node.DefineFunction):
                return True
        return False


class Translator:
    def __init__(self, cc: 'Compiler'):
        self.cc = cc
        self.num_vars = 0
        self.locals = {'_', 'ARGS'}
        self.frames = []

    def translate(self, node):
        code = []
        indent = 2  # see py_template
        cur_loc = '(0,0)'
        text_buf = []

        for elem in _flatten(self.emit(node)):
            if isinstance(elem, Node.Text):
                text_buf.append(elem.text or '')
                continue

            if text_buf:
                s = ''.join(text_buf)
                if s:
                    code.append((C.PY_INDENT * indent) + f'_ENV.echo({s!r})')
                text_buf = []

            if isinstance(elem, Node.Location):
                new_loc = _parens(f'{elem.loc.path_index},{elem.loc.line_num}')
                if new_loc != cur_loc:
                    code.append(C.PY_MARKER + _path_line(elem.loc.path, elem.loc.line_num))
                    cur_loc = new_loc
                continue

            if elem == C.PY_BEGIN:
                indent += 1
                continue

            if elem == C.PY_END:
                indent -= 1
                continue

            if elem.strip():
                code.append((C.PY_INDENT * indent) + elem.replace('$loc$', cur_loc))

        s = ''.join(text_buf)
        if s:
            code.append((C.PY_INDENT * indent) + f'_ENV.echo({s!r})')

        py = C.PY_TEMPLATE
        py = py.replace('$name$', self.cc.options.name)
        py = py.replace('$paths$', repr(self.cc.buf.paths))
        py = py.replace('$code$', C.NL + C.NL.join(code) + C.NL)
        py = py.replace('$loc$', cur_loc)

        return py

    def var(self):
        self.num_vars += 1
        return '_' + str(self.num_vars)

    def enter_frame(self):
        self.frames.append(self.locals)
        self.locals = set(self.locals)

    def leave_frame(self):
        self.locals = self.frames.pop()

    def emit(self, node):
        return node.emit(self)

    def emit_try(self, block, fallback=None, mute=False):
        exc = self.var()

        if mute:
            err = fallback or 'pass'
        else:
            err = [f'_ENV.error({exc}, $loc$)', fallback or '']

        return [
            'try:',
            C.PY_BEGIN,
            block,
            C.PY_END,
            f'except Exception as {exc}:',
            C.PY_BEGIN,
            err,
            C.PY_END
        ]

    def emit_assign(self, var, value, fallback='None', mute=False):
        return self.emit_try(
            f'{var} = {value}',
            f'{var} = {fallback}',
            mute=mute
        )

    def emit_echo(self, value, mute=False):
        res = self.var()
        code = [
            f'{res} = {value}',
            f'_ENV.echo({res})'
        ]
        return self.emit_try(code, mute=mute)

    def emit_left_binary_op(self, node):
        # a + b + c => ((a + b) + c)
        code = self.emit(node.subject)
        for op, other in node.pairs:
            code = _parens(f'{code} {op} {self.emit(other)}')
        return code

    def emit_right_binary_op(self, node):
        # a ** b ** c => (a ** (b ** c))
        code = self.emit(node.subject)
        for op, other in node.pairs:
            code = f'{code} {op} ({self.emit(other)}'
        code += ')' * len(node.pairs)
        return _parens(code)

    def emit_comp_binary_op(self, node):
        # a < b < c => (a) < (b) < (c)
        codes = [self.emit(node.subject)]
        for op, other in node.pairs:
            codes.append(op)
            codes.append(_parens(self.emit(other)))
        return ' '.join(codes)

    def emit_unary_op(self, node):
        code = self.emit(node.subject)
        for op in node.ops:
            space = ' ' if op.isalnum() else ''
            code = _parens(f'{op}{space}{code}')
        return code


class Compiler:
    def __init__(self, engine, options):
        self.engine = engine
        self.buf = Buffer()

        opts = dict(C.DEFAULT_OPTIONS)
        opts.update(options or {})
        self.options = Data(**opts)

    def load(self, basepath, path, loc):
        try:
            if callable(self.options.loader):
                return self.options.loader(basepath, path)
            if not os.path.isabs(path) and basepath:
                path = os.path.abspath(os.path.join(os.path.dirname(basepath), path))
            with open(path, 'rt', encoding='utf8') as fp:
                text = fp.read()
            return text, path
        except OSError as exc:
            raise CompileError(f'cannot load {path!r}: {exc.args[1]}', loc)

    def parse(self):
        return TemplateParser(self).parse()

    def translate(self, node):
        return Translator(self).translate(node)

    def compile(self, python):
        local_vars = {}
        exec(python, {}, local_vars)
        return local_vars[self.options.name]


def _dedent(lines):
    ind = 100_000
    for ln in lines:
        n = len(ln.lstrip())
        if n > 0:
            ind = min(ind, len(ln) - n)
    return [ln[ind:] for ln in lines]


def _indent(lines, size=4):
    sp = ' ' * size
    return [sp + ln for ln in lines]


def _flatten(ls):
    if not isinstance(ls, (list, tuple)):
        yield ls
        return

    for item in ls:
        if not isinstance(item, (list, tuple)):
            yield item
        else:
            yield from _flatten(item)


def _path_line(path, line):
    s = repr(path)[1:-1]
    return s + ':' + str(line)


def _parens(s):
    return '(' + s + ')'


def _unquote(s):
    return s.strip('\'\" ')


def _cut(s, maxlen):
    if len(s) <= maxlen:
        return s
    return s[:maxlen] + '...'


_comma = ','.join

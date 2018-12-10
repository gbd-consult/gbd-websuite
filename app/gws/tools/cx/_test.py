import re
import os
import pytest

from . import compiler, template


def _nows(s):
    return re.sub(r'\s+', '', s.strip())


def _render(txt, ctx):
    ## print(compiler.translate(txt, base=template.Template))

    cls = compiler.compile_string(txt, base=template.Template)
    tpl = cls()
    out = tpl.render(ctx)

    ## print(tpl.errors)

    return out, tpl.errors


# Basic syntax

def test_render_no_commands():
    t = """aa bb cc dd"""
    s, err = _render(t, {})
    assert s == t
    assert not err


def test_simple_property():
    t = '>{aa}<'
    d = {'aa': 123}
    s, err = _render(t, d)
    assert s == '>123<'
    assert not err


def test_nested_property():
    t = '>{aa["bb"][0]["cc"]}<'
    d = {'aa': {'bb': [{'cc': 123}]}}
    s, err = _render(t, d)
    assert s == '>123<'
    assert not err


def test_expression():
    t = '>{aa["bb"] + 7 * 1}<'
    d = {'aa': {'bb': 123}}
    s, err = _render(t, d)
    assert s == '>130<'
    assert not err


# @let

def test_let_expression():
    tpl = """
        @let aa 123
        >{aa + 1}<
    """
    s, err = _render(tpl, {})
    assert _nows(s) == '>124<'

    tpl = """
        @let aa 'abc'
        @let bb '[' + aa + ']'
        >{bb}<
    """
    s, err = _render(tpl, {})
    assert _nows(s) == '>[abc]<'


def test_let_block():
    tpl = """
        @let aa
            abc
            def
        @end
        >{aa}<
        
    """
    s, err = _render(tpl, {})
    assert _nows(s) == '>abcdef<'


def test_nested_let_block():
    tpl = """
        @let aa
            abc
            @let bb
                uwv
            @end
            def
        @end
        >{aa}<
        >{bb}<
        
    """
    s, err = _render(tpl, {})
    assert _nows(s) == '>abcdef<>uwv<'


# @if

def test_if():
    tpl = """
        @if 2 > 1
            123
        @end
    """
    s, err = _render(tpl, {})
    assert _nows(s) == '123'

    tpl = """
        @if aa > 1
            123
        @end
    """
    s, err = _render(tpl, {'aa': 2})
    assert _nows(s) == '123'


def test_if_else():
    tpl = """
        @if aa > 1
            123
        @else
            456
        @end
    """
    s, err = _render(tpl, {'aa': 0})
    assert _nows(s) == '456'


def test_if_elif():
    tpl = """
        @if aa > 10
            >10
        @elif aa > 5
            >5
        @else
            <=5
        @end
    """
    s, err = _render(tpl, {'aa': 20})
    assert _nows(s) == '>10'
    s, err = _render(tpl, {'aa': 8})
    assert _nows(s) == '>5'
    s, err = _render(tpl, {'aa': 4})
    assert _nows(s) == '<=5'


def test_nested_if():
    tpl = """
        @if aa > 5
            >5
            @if aa > 10
                >10
                @if aa > 20
                    >20
                @end
            @else
                <10
            @end
        @elif aa > 2
            >2
        @else
            <=2
            @if aa > 1
                =2
            @else
                =1
            @end
        @end
    """
    s, err = _render(tpl, {'aa': 25})
    assert _nows(s) == '>5>10>20'

    s, err = _render(tpl, {'aa': 15})
    assert _nows(s) == '>5>10'

    s, err = _render(tpl, {'aa': 5})
    assert _nows(s) == '>2'

    s, err = _render(tpl, {'aa': 2})
    assert _nows(s) == '<=2=2'

    s, err = _render(tpl, {'aa': 1})
    assert _nows(s) == '<=2=1'


# @each

def test_each():
    d = {'lst': ['aa', 'bb', 'cc'], 'dct': {'xx': 11, 'yy': 22, 'zz': 33}}

    tpl = """
        @each lst
            *
        @end
    """
    s, err = _render(tpl, d)
    assert _nows(s) == '***'

    tpl = """
        @each lst as v
            {v}/
        @end
        ...
        @each dct as v
            {v}/
        @end
    """
    s, err = _render(tpl, d)
    assert _nows(s) == 'aa/bb/cc/...xx/yy/zz/'

    tpl = """
        @each lst as k, v
            {k}_{v}/
        @end
        ...
        @each dct as k, v
            {k}_{v}/
        @end
    """
    s, err = _render(tpl, d)
    assert _nows(s) == '0_aa/1_bb/2_cc/...xx_11/yy_22/zz_33/'


def test_each_with_extras():
    d = {'lst': ['aa', 'bb', 'cc']}

    tpl = """
        @each lst as v (index)
            {index}_{v}/
        @end
        ...
        @each lst as v (index, length)
            {index}_{v}_{length}/
        @end
    """
    s, err = _render(tpl, d)
    assert _nows(s) == '1_aa/2_bb/3_cc/...1_aa_3/2_bb_3/3_cc_3/'

def test_nested_each():
    d = {'lst': ['aa', 'bb', 'cc']}

    tpl = """
        @each lst as v (index)
            {index}_{v}/
        @end
        ...
        @each lst as v (index, length)
            {index}_{v}_{length}/
        @end
    """
    s, err = _render(tpl, d)
    assert _nows(s) == '1_aa/2_bb/3_cc/...1_aa_3/2_bb_3/3_cc_3/'


    # def command_pragma(self, arg):
    # def command_text(self, arg):
    # def command_def(self, arg):
    # def command_filter(self, arg):
    # def command_command(self, arg):
    # def command_code(self, arg):
    # def command_each(self, arg):
    # def command_with(self, arg):
    # def command_include(self, arg):






#
# def test_loop_nokey():
#     tpl = """
#         @each ['resh', 'shin', 'tav'] as e
#             *
#         @end
#     """
#     s, err =_render(tpl, {})
#     assert _nows(s) == '***'
#
#
# def test_loop_key1():
#     tpl = """
#         @each ['resh', 'shin', 'tav'] as e
#             {e}!
#         @end
#     """
#     s, err =_render(tpl, {})
#     assert _nows(s) == 'resh!shin!tav!'
#
#
# def test_loop_key2():
#     tpl = """
#         @each ['resh', 'shin', 'tav'] as n, e
#             {n}={e}!
#         @end
#     """
#     s, err =_render(tpl, {})
#     print(err)
#     assert _nows(s) == '1=resh!2=shin!3=tav!'
#
#
# def test_loop_empty():
#     tpl = """
#         >
#         @each [] as e
#             {e}
#         @end
#         <
#     """
#     s, err =_render(tpl, {})
#     assert _nows(s) == '><'
#
#
# def test_loop_empty_else():
#     tpl = """
#         >
#         @each [] as e
#             {e}
#         @else
#             EMPTY!
#         @end
#         <
#     """
#     s, err =_render(tpl, {})
#     assert _nows(s) == '>EMPTY!<'
#
#
# def test_with_empty():
#     tpl = """
#         >
#         @with {aa}
#             123
#         @end
#         <
#     """
#     s, err =_render(tpl, {})
#     assert _nows(s) == '><'
#
#     s, err =_render(tpl, {'aa': ''})
#     assert _nows(s) == '><'
#
#     s, err =_render(tpl, {'aa': {}})
#     assert _nows(s) == '><'
#
#     s, err =_render(tpl, {'aa': ''})
#     assert _nows(s) == '><'
#
#
# def test_with_not_empty():
#     tpl = """
#         >
#         @with {aa}
#             123
#         @end
#         <
#     """
#     s, err =_render(tpl, {'aa': 1})
#     assert _nows(s) == '>123<'
#
#     s, err =_render(tpl, {'aa': 0})
#     assert _nows(s) == '>123<'
#
#
# def test_with_ref():
#     tpl = """
#         >
#         @with {aa} as a
#             {a.bb}
#         @end
#         <
#     """
#     s, err =_render(tpl, {'aa': {'bb': 456}})
#     assert _nows(s) == '>456<'
#
#
# def test_inline_code():
#     tpl = """
#         >
#         @@  lamed = 'LAMED'
#         >
#         `lamed + '!' `
#         >
#     """
#
#     s, err =_render(tpl, {})
#     assert _nows(s) == '>>LAMED!>'
#
#
# def test_code_block():
#     tpl = """
#         >
#         @code
#             a = 5
#             if a > 1:
#                 a += 10
#         @end
#         `a + 200`
#         <
#     """
#
#     s, err =_render(tpl, {})
#     assert _nows(s) == '>215<'
#
#
# BASE_PATH = '/tmp/cxtest'
#
#
# def _wfile(path, tpl):
#     with open(BASE_PATH + '/' + path, 'wt') as fp:
#         fp.write(tpl)
#
#
# def test_include():
#     os.system('rm -fr %s' % BASE_PATH)
#     os.system('mkdir -p %s/aa' % BASE_PATH)
#     os.system('mkdir -p %s/bb/gimel' % BASE_PATH)
#
#     _wfile('tav1', 'TAV-1')
#     _wfile('aa/alef1', 'ALEF-1')
#
#     _wfile('bb/bet1', '''
#             BET-1
#             |
#             @include ../aa/alef1
#             |
#             @include bet2
#             |
#             @include gimel/gimel1
#         ''')
#
#     _wfile('bb/bet2', 'BET-2')
#     _wfile('bb/bet3', 'BET-3')
#     _wfile('bb/gimel/gimel1', '''
#         GIMEL-1
#         |
#         @include ../bet3
#     ''')
#
#     s, err =_render.from_path(BASE_PATH + '/bb/bet1', {})
#     assert _nows(s) == 'BET-1|ALEF-1|BET-2|GIMEL-1|BET-3'
#
#     os.system('rm -fr %s' % BASE_PATH)
#
#
# def test_def():
#     tpl = """
#         @def aa(a, b)
#             ( {@b} _ {@a} )
#             `a.upper() + '/' + b.upper()`
#         @end
#
#         |
#         `aa({lamed}, {mem})`
#         |
#         `aa({nun}, {samekh})`
#         |
#     """
#
#     s, err =_render(tpl, {
#         'lamed': 'lamed1',
#         'mem': 'mem1',
#         'nun': 'nun1',
#         'samekh': 'samekh1',
#     })
#     assert _nows(s) == '|(mem1_lamed1)LAMED1/MEM1|(samekh1_nun1)NUN1/SAMEKH1|'
#
#
# def test_def_noargs():
#     tpl = """
#         @def aa
#             bb!
#         @end
#         |
#         `aa()`
#         |
#     """
#
#     s, err =_render(tpl)
#     assert _nows(s) == '|bb!|'
#
#
# def test_def_explicit_return():
#     t = """
#         @def aa(a, b)
#             @@ return a + b
#         @end
#
#         |
#         `aa(10, 200)`
#         |
#     """
#     print(compiler.translate(t))
#     s, err =_render(t)
#     assert _nows(s) == '|210|'
#
#
# def test_filter_simple():
#     t = '''
#         >{aa upper}<
#     '''
#     s, err =_render(t, {'aa': 'tav'})
#     assert _nows(s) == '>TAV<'
#
#
# def test_filter_args():
#     t = '''
#         >{aa slice(1,-1)}<
#     '''
#     s, err =_render(t, {'aa': '?tav?'})
#     assert _nows(s) == '>tav<'
#
#
# def test_filter_cascade():
#     t = '''
#         >{aa upper slice(1,-1)}<
#     '''
#     s, err =_render(t, {'aa': '?tav?'})
#     assert _nows(s) == '>TAV<'
#
#
# def test_filter_custom():
#     t = '''
#         @def rev(arg)
#             `''.join(reversed({@arg}))`
#         @end
#
#         >{aa rev}<
#     '''
#     s, err =_render(t, {'aa': 'tav'})
#     assert _nows(s) == '>vat<'
#
#
# def test_errors():
#     with pytest.raises(compiler.Error, match='ERROR_SYNTAX'):
#         _render(' `1+` ')
#     with pytest.raises(compiler.Error, match='ERROR_COMMAND'):
#         _render('''
#             @foo
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_EOF'):
#         _render('''
#             @if 1
#                 blah
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_KEY'):
#         _render('''
#            {...}
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_ASSIGN'):
#         _render('''
#            @let [foo] 1
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_VALUE'):
#         _render('''
#            {|foo}
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_FILTER_SYNTAX'):
#         _render('''
#            {foo | ???}
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_NO_FILTER'):
#         _render('''
#            {foo | blah}
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_ELSE'):
#         _render('''
#             @if 1
#             @else 3
#             @else 3
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_DEF'):
#         _render('''
#            @def ???
#         ''')
#     with pytest.raises(compiler.Error, match='ERROR_FILE'):
#         _render('''
#            @include nothere
#         ''')

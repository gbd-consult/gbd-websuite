from .engine import Engine
from .compiler import Compiler, CompileError

_DefaultEngine = Engine()


def engine():
    return _DefaultEngine


def parse(text, **options):
    return _DefaultEngine.parse(text, **options)


def parse_path(path, **options):
    return _DefaultEngine.parse_path(path, **options)


def translate(text, **options):
    return _DefaultEngine.translate(text, **options)


def translate_path(path, **options):
    return _DefaultEngine.translate_path(path, **options)


def compile(text, **options):
    return _DefaultEngine.compile(text, **options)


def compile_path(path, **options):
    return _DefaultEngine.compile_path(path, **options)


def call(template_fn, args=None, error=None):
    return _DefaultEngine.call(template_fn, args, error)


def render(text, args=None, error=None, **options):
    return _DefaultEngine.render(text, args, error, **options)


def render_path(path, args=None, error=None, **options):
    return _DefaultEngine.render_path(path, args, error, **options)

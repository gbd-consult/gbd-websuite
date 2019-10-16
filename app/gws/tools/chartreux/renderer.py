from . import compiler, runtime as rt


def call(
        template,
        context=None,
        runtime=None,
        error=None,
):
    return template(runtime or rt.Runtime, context, error)


def render(
        text,

        context=None,
        error=None,
        runtime=None,

        filter=None,
        finder=None,
        globals=None,
        name=None,
        path=None,
        syntax=None,
):
    template = compiler.compile(
        text,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        path=path,
        syntax=syntax,
    )
    return call(
        template,
        context=context,
        runtime=runtime,
        error=error,
    )


def render_path(
        path,

        context=None,
        error=None,
        runtime=None,

        filter=None,
        finder=None,
        globals=None,
        name=None,
        syntax=None,
):
    template = compiler.compile_path(
        path,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        syntax=syntax,
    )
    return call(
        template,
        context=context,
        runtime=runtime,
        error=error,
    )

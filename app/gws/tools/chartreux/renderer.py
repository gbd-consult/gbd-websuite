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

        syntax=None,
        filter=None,
        globals=None,
        name=None,
        path=None,
        silent=False,
):
    template = compiler.compile(
        text,
        syntax=syntax,
        filter=filter,
        globals=globals,
        name=name,
        path=path,
        silent=silent,
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

        syntax=None,
        filter=None,
        globals=None,
        name=None,
        silent=False,
):
    template = compiler.compile_path(
        path,
        syntax=syntax,
        filter=filter,
        globals=globals,
        name=name,
        silent=silent,
    )
    return call(
        template,
        context=context,
        runtime=runtime,
        error=error,
    )

from . import compiler, runtime as rt



_DefaultRuntime = rt.Runtime()

def call(
        template,
        context=None,
        runtime=None,
        error=None,
):
    return template(runtime or _DefaultRuntime, context, error)


def render(
        text,

        context=None,
        error=None,
        runtime=None,

        commands=None,
        filter=None,
        finder=None,
        globals=None,
        name=None,
        path=None,
        strip=None,
        syntax=None,
):
    template = compiler.compile(
        text,
        commands=commands,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        path=path,
        strip=strip,
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

        commands=None,
        filter=None,
        finder=None,
        globals=None,
        name=None,
        strip=None,
        syntax=None,
):
    template = compiler.compile_path(
        path,
        commands=commands,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        strip=strip,
        syntax=syntax,
    )
    return call(
        template,
        context=context,
        runtime=runtime,
        error=error,
    )

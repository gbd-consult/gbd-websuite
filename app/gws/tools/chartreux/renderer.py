from . import compiler, runtime as rt


def call(
        template,
        context=None,
        runtime=None,
        error=None,
):
    return template(runtime or rt.DefaultRuntime, context, error)


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
        strip=None,
        syntax=None,
        commands=None,
):
    template = compiler.compile(
        text,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        path=path,
        strip=strip,
        syntax=syntax,
        commands=commands,
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
        strip=None,
        syntax=None,
        commands=None,
):
    template = compiler.compile_path(
        path,
        filter=filter,
        finder=finder,
        globals=globals,
        name=name,
        strip=strip,
        syntax=syntax,
        commands=commands,
    )
    return call(
        template,
        context=context,
        runtime=runtime,
        error=error,
    )

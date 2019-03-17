from . import compiler, runtime as rt


def render(
        template,
        context=None,
        warn=None,
        runtime=None,
):
    return template(runtime or rt.Runtime, context, warn)


def render_text(
        text,
        context=None,
        warn=None,
        filter=None,
        extern=None,
        name='render',
        path=None,
        runtime=None,
        silent=False,
):
    template = compiler.compile(
        text,
        filter=filter,
        extern=extern,
        name=name,
        path=path,
        silent=silent,
    )
    return render(
        template,
        context=context,
        warn=warn,
        runtime=runtime,
    )


def render_path(
        path,
        context=None,
        warn=None,
        filter=None,
        extern=None,
        name='render',
        runtime=None,
        silent=False,
):
    with open(path) as fp:
        source = fp.read()
    return render_text(
        source,
        context=context,
        warn=warn,
        filter=filter,
        extern=extern,
        name=name,
        path=path,
        runtime=runtime,
        silent=silent,
    )

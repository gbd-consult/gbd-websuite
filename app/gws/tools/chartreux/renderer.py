from . import compiler, runtime as rt


def render(
        template,
        context=None,
        runtime=None,
        warn=None,
):
    return template(runtime or rt.Runtime, context, warn)


def render_text(
        text,
        context=None,
        filter=None,
        globals=None,
        name='render',
        path=None,
        runtime=None,
        silent=False,
        warn=None,
):
    template = compiler.compile(
        text,
        filter=filter,
        globals=globals,
        name=name,
        path=path,
        silent=silent,
    )
    return render(
        template,
        context=context,
        runtime=runtime,
        warn=warn,
    )


def render_path(
        path,
        context=None,
        filter=None,
        globals=None,
        name='render',
        runtime=None,
        silent=False,
        warn=None,
):
    with open(path) as fp:
        source = fp.read()
    return render_text(
        source,
        context=context,
        filter=filter,
        globals=globals,
        name=name,
        path=path,
        runtime=runtime,
        silent=silent,
        warn=warn,
    )

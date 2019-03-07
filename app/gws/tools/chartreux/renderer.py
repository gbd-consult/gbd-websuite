from . import compiler, runtime as rt


def render(
        template,
        context=None,
        errors=None,
        runtime=None,
):
    err = []
    out = template(runtime or rt.Runtime, context, err)
    if errors is not None:
        errors.extend(err)
    return out


def render_text(
        text,
        context=None,
        errors=None,
        filter=None,
        globals=None,
        name='render',
        path=None,
        runtime=None,
        silent=False,
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
        errors=errors,
        runtime=runtime,
    )


def render_path(
        path,
        context=None,
        errors=None,
        filter=None,
        globals=None,
        name='render',
        runtime=None,
        silent=False,
):
    with open(path) as fp:
        source = fp.read()
    return render_text(
        source,
        context=context,
        errors=errors,
        filter=filter,
        globals=globals,
        name=name,
        path=path,
        runtime=runtime,
        silent=silent,
    )

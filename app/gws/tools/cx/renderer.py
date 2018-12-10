from . import compiler, template


def _render(text, context, errors, opts):
    klass = compiler.compile(text, **opts)
    tpl = klass()
    out = tpl.render(context, **opts)
    if isinstance(errors, list):
        errors.extend(tpl.errors)
    return out


def render(text, context, errors=None, **opts):
    opts = _check_opts(opts)
    return _render(text, context, errors, opts)


def render_path(path, context, errors=None, **opts):
    with open(path) as fp:
        text = fp.read()
    opts = _check_opts(opts)
    opts['path'] = path
    return _render(text, context, errors, opts)


def _check_opts(opts):
    opts = opts or {}
    if 'base' not in opts:
        opts['base'] = template.Template
    return opts

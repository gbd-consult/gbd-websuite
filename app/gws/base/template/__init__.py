"""Templates.

Templates generate content from data: HTML pages, feature descriptions, print
layouts and other output. A template is identified by its ``subject``, which
names the purpose it serves, for example ``feature.description``,
``layer.description``, ``project.description``, ``project.home`` or
``application.home``. Concrete template types are plugins: ``html``, ``text``,
``map``, ``py`` and ``qgis``.

Submodules:

- ``core``: the base class for templates (``gws.Template``), with the common
  ``Config`` options (``subject``, ``title``, ``mimeTypes``, ``pageSize``,
  ``mapSize``, ``pageMargin``). If not configured, the map size defaults to
  50 x 50 mm and the page size to A4 portrait. It prepares the template
  arguments: the application, the subject, the locale with date, time and number
  formatters, and the ``STRINGS`` dictionary with localized default strings.
- ``manager``: the template manager (``gws.TemplateManager``), which finds
  templates by subject and creates templates from file paths.

Templates are configured in the ``templates`` list of objects such as layers,
models, finders, projects and the application. The manager looks up a
template by subject in a list of objects, in the given order, and falls back to
the application templates. Templates the user may not use, or which cannot
produce the requested MIME type, are skipped. A template file in an assets
directory is turned into a template by its extension (``.cx.html``, ``.cx.py``,
``.qgs`` and others, see ``manager.TEMPLATE_TYPES``).

Example::

    {
        templates [
            { subject "feature.title" type "html" text "{{name}}" }
            { subject "feature.description" type "html" path "/data/templates/description.cx.html" }
        ]
    }

Example::

    tpl = root.app.templateMgr.find_template('feature.description', where=[layer, project], user=user)
    if tpl:
        res = tpl.render(gws.TemplateRenderInput(args={'feature': feature}, user=user))
"""

from .core import Config, Object, Props
from . import manager


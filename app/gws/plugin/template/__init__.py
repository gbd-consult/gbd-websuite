"""Template types.

This package contains the template plugins. Templates format feature
descriptions, labels, pages and print layouts; they implement the
``gws.Template`` interface and extend ``gws.base.template.Object``, which
provides the common options (``subject``, ``title``, ``mimeTypes``,
``pageSize``, ``mapSize``, ``pageMargin``) and the template arguments.

Subpackages
-----------

- ``html`` - templates in the Jump language that produce HTML, PDF or PNG
  output, with commands for maps, legends, page setup, headers and footers.
  This is the main template type, used for feature formatting and printing.
- ``text`` - Jump templates for plain text output, without the HTML
  commands.
- ``map`` - a template that renders only the map, as HTML, PDF or PNG.
- ``py`` - a template implemented as a Python module with a ``main``
  function.

Print templates based on QGIS layouts are in ``gws.plugin.qgis.template``.

Example::

    templates+ {
        subject "feature.description"
        type "html"
        path "/data/templates/description.cx.html"
    }

    printers+ {
        template {
            type "html"
            path "/data/templates/print.cx.html"
            mimeTypes [ "application/pdf" ]
        }
    }
"""

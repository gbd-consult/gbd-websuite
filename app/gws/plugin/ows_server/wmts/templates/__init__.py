"""Default templates of the WMTS service.

The templates are ``py`` templates. Each module defines a ``main`` function
that takes the template arguments (``gws.base.ows.server.TemplateArgs``) and
returns the response, mostly built with ``gws.base.ows.server.templatelib``.

Templates:

- ``getCapabilities.cx.py``: the capabilities document (``ows.GetCapabilities``),
  including the tile matrix sets.
"""

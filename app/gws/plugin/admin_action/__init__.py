"""Administration tools.

The ``admin`` action provides html pages for administrators. Both commands
require the ``admin`` role and raise ``gws.ForbiddenError`` otherwise.

Submodules:

- ``action``: the ``admin`` action with the ``adminMapCache`` and ``adminInspector`` GET commands.
- ``inspector``: the object inspector, a page to browse the configured object tree and
  search it for property values.
- ``mapcache``: the cache viewer, a page that shows the tile caches, their levels and
  cached tiles on a map.

Each tool renders a page from its ``page.cx.html`` template and serves its own
assets (``page.js``, ``page.css``) through the same command, using the ``path``
parameter.

Example::

    actions+ { type admin }

The tools are then available at ``/_/adminInspector`` and ``/_/adminMapCache``.
"""

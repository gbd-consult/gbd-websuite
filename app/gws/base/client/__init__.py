"""Browser client configuration and bundles.

The client object describes the browser client of the application or a
project: a list of UI elements, identified by tags such as ``Toolbar.Print``
or ``Sidebar.Layers``, and a dict of client options. Both are sent to the
browser as client props. Elements can have access rules, so different users can
see different elements.

The application has a default client, which is used by projects without a
client configuration. A project client inherits the
application options (merged with its own) and the application elements. A
project can replace the element list with ``elements``, or modify the inherited
one with ``addElements`` (placed ``before`` or ``after`` an existing tag, or
appended) and ``removeElements``.

Submodules:

- ``core``: the client ``Object``, the ``Element`` node and their configs and props.
- ``bundles``: serves the JavaScript and CSS bundles built by the JS bundler
  (``app/js/helpers/builder.js``), with the UI strings for the client locale
  and the CSS for the requested theme.

Example::

    client {
        elements [
            { tag "Sidebar.Layers" }
            { tag "Toolbar.Print" access "allow user, deny all" }
        ]
        options {
            sidebarActiveTab "Sidebar.Layers"
            sidebarVisible true
        }
    }

    projects [
        {
            uid "project_1"
            client.addElements [
                { tag "Sidebar.Edit" after "Sidebar.Layers" }
            ]
            client.removeElements [
                { tag "Toolbar.Print" }
            ]
        }
    ]
"""

from .core import Object, Props, Config

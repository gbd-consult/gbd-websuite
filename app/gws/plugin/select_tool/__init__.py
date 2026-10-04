"""Feature selection tool.

This plugin provides the ``select`` action and the client tools for
selecting features on the map, by a click or by a drawn polygon. Selected
features are listed in a sidebar tab, where the selection can be cleared,
saved and loaded again.

Submodules
----------

- ``action`` - the ``select`` action. It passes the click ``tolerance`` to
  the client and, if a ``storage`` is configured, handles the requests for
  saving and loading selections (category ``Select``).

The selection itself runs in the client and uses the regular search; the
client elements are ``Sidebar.Select``, ``Toolbar.Select`` and
``Toolbar.Select.Draw``.

Example::

    actions+ {
        type "select"
        tolerance "10px"
        storage {
            permissions {
                read "allow all"
                write "allow all"
                create "allow all"
            }
        }
    }

    client.addElements+ { tag "Sidebar.Select" }
    client.addElements+ { tag "Toolbar.Select" }
    client.addElements+ { tag "Toolbar.Select.Draw" }
"""

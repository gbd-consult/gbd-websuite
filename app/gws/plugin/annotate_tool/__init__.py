"""Annotate tool.

Client tool for drawing shapes on the map (points, lines, polygons, circles,
boxes) and labeling them with measurements, such as coordinates, length,
perimeter, area or radius. The annotations are listed in a sidebar, where
their labels and styles can be edited. If a storage is configured, they can be
saved on the server and loaded again.

Submodules
----------

- ``action`` - the ``annotate`` action. It passes the label templates
  (``labels``) and the storage settings to the client and implements the API
  command ``annotateStorage``, which reads and writes saved annotations.
- ``js`` - the client part: the elements ``Sidebar.Annotate``,
  ``Toolbar.Annotate.Draw`` and ``Task.Annotate``.

The ``permissions`` of the storage define who may read, overwrite and create
saved annotations.

Example::

    actions+ {
        type "annotate"
        storage {
            permissions {
                read "allow all"
                write "allow all"
                create "allow all"
            }
        }
    }

    client.addElements+ { tag "Sidebar.Annotate" }
    client.addElements+ { tag "Toolbar.Annotate.Draw" }
    client.addElements+ { tag "Task.Annotate" }
"""

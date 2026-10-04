"""Model widgets.

Each subpackage implements one widget type (``gws.ext.object.modelWidget``).
A widget tells the client how to display and edit a field. The server side only
passes the widget options to the client in the widget props; the widgets
themselves are implemented in the client, mostly in the ``index.tsx`` file of
the widget package.

When a field has no ``widget`` config, it creates a default widget for its type,
e.g. ``input`` for text fields or ``featureSelect`` for related features.

- ``date``: input for dates.
- ``feature_list``: list of related features, with buttons to create, link, edit, unlink and delete.
- ``feature_select``: drop-down list of related features.
- ``feature_suggest``: input that suggests related features.
- ``file``: upload and download of a file.
- ``file_list``: list of related features that hold files.
- ``float``: input for decimal numbers.
- ``geometry``: buttons to draw or edit a geometry.
- ``hidden``: the field is not shown.
- ``input``: single-line text input.
- ``integer``: input for whole numbers.
- ``password``: masked password input.
- ``select``: drop-down list with a fixed set of values.
- ``textarea``: multi-line text input.
- ``toggle``: checkbox or radio button.

Example::

    fields+ {
        name "kind"
        type "text"
        widget {
            type "select"
            items [
                { value "a" text "Type A" }
                { value "b" text "Type B" }
            ]
        }
    }
"""

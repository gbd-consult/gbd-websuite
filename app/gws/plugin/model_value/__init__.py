"""Model values.

Each subpackage implements one value type (``gws.ext.object.modelValue``).
Values are configured in the ``values`` list of a model field and compute the
field value when a feature is read, created or updated (``forRead``,
``forCreate``, ``forUpdate``). With ``isDefault``, the value is used only when
no other value is provided. Values also serve as bounds for range validators.

- ``current_timestamp``: the current date and time.
- ``current_user``: the login name of the current user, or a formatted string.
- ``expression``: the result of a Python expression.
- ``format``: a format string applied to the feature attributes.
- ``static``: a fixed value.

Example::

    fields+ {
        name "updated_by"
        type "text"
        values+ { type "currentUser" forRead false forUpdate true }
    }

    fields+ {
        name "status"
        type "text"
        values+ { type "static" value "new" isDefault true }
    }
"""

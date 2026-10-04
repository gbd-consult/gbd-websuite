"""Model validators.

Each subpackage implements one validator type (``gws.ext.object.modelValidator``).
Validators are configured in the ``validators`` list of a model field and check
the field value of a feature before it is created or updated. A validator returns
``False`` when the value is invalid; the field then adds the validator's message
to the feature errors.

Every field runs a ``notEmpty`` and a ``format`` validator first; configured
validators of these types replace the default ones. An empty value stops the
validation, and is an error only if the field is required. A value that could
not be parsed is an error. The other validators run only after these checks
pass, and the first one that fails stops the validation.

- ``date_range``: the value is a date within the given bounds.
- ``format``: the value could be parsed for the field type.
- ``not_empty``: the value is not empty.
- ``number_range``: the value is a number within the given bounds.
- ``regex``: the value is a string that matches a regular expression.

The bounds of ``date_range`` and ``number_range`` are model values
(see ``gws.plugin.model_value``), so they can be static or computed.

Example::

    fields+ {
        name "code"
        type "text"
        isRequired true
        validators+ { type "regex" regex "^[A-Z]{2}[0-9]+$" }
    }

    fields+ {
        name "built"
        type "date"
        validators+ {
            type "dateRange"
            min { type "static" value "1900-01-01" }
            max { type "currentTimestamp" }
        }
    }
"""

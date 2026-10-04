"""Model field types.

Each subpackage implements one field type for models (``gws.ext.object.modelField``).
A field maps one attribute of a feature to a column (or a relationship) in the
data source, converts values between the database, Python and the client, and
creates a default widget when none is configured.

Scalar fields, based on ``gws.base.model.scalar_field``:

- ``bool``: boolean values, ``toggle`` widget.
- ``date``: dates, transferred as ISO strings, ``date`` widget.
- ``datetime``: date and time values, transferred as ISO strings, ``input`` widget.
- ``float``: floating-point numbers, ``float`` widget.
- ``integer``: integer numbers, ``integer`` widget.
- ``text``: strings, with optional keyword search, ``input`` widget.
- ``time``: time values, transferred as ISO strings, ``input`` widget.
- ``geometry``: geometries, used for spatial searches, ``geometry`` widget.

Other fields, based on ``gws.base.model.field``:

- ``file``: files stored in a database column, with download and preview URLs.

Related fields, based on ``gws.base.model.related_field``, link features of
database models:

- ``related_feature``: M:1, the value is the parent feature.
- ``related_feature_list``: 1:M, the value is the list of child features.
- ``related_multi_feature_list``: 1:M with several child models.
- ``related_linked_feature_list``: M:N via a link table.

Example::

    models+ {
        type "postgres"
        tableName "edit.poi"
        fields+ { name "id" type "integer" isPrimaryKey true }
        fields+ { name "name" type "text" textSearch { type "any" } }
        fields+ { name "geom" type "geometry" }
        fields+ {
            name "category"
            type "relatedFeature"
            fromColumn "category_id"
            toModel "model_category"
            widget.type "featureSelect"
        }
    }
"""

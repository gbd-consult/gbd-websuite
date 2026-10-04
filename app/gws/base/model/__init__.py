"""Data models.

A data model, or simply model, is an object that deals with features from
external sources, like database tables, shape files, GML responses, etc. The
job of a model is to read features from the source and convert them to a form
suitable for the client. An editable model can also accept features back from
the client, parse and validate them, and store them in the source.

This package provides the base classes for models and their components. Concrete
models (e.g. ``postgres``, ``qgis``, ``geojson``) live in ``gws.base.database`` and
in plugins, concrete fields, values, validators and widgets live in the
``gws.plugin.model_field``, ``gws.plugin.model_value``, ``gws.plugin.model_validator``
and ``gws.plugin.model_widget`` packages.

Submodules
----------

- ``core`` - the base model class (`Object`) with the model configuration protocol,
  props and the conversion between features and props.
- ``default_model`` - the ``default`` model, used when no model is configured. It copies
  attributes between records, features and props as they are.
- ``manager`` - the model manager (``root.app.modelMgr``), which looks up models
  by uid or by the objects they belong to, and lists editable models. Editable models
  are collected from the project layers, the project and the application, together with
  their related models the user can read.
- ``field`` - the base model field class, with flags, values, validators, widget and validation.
- ``scalar_field`` - the base class for fields that map to one source attribute (column).
- ``related_field`` - the base class for fields that link features of other models,
  with the relationship description and database helpers.
- ``value`` - the base class for value objects.
- ``validator`` - the base class for validator objects.
- ``widget`` - the base class for client widgets.
- ``util`` - helpers to create model contexts and to iterate over related features.

Features
--------

A feature is a collection of named attributes. One of these attributes can
act as a unique ID (``uid``), and another one can be the feature
geometry (``shape``). ``uid`` is required for editable models, ``shape`` is always optional.

There are three kinds of objects that represent features:

The feature object (`gws.Feature`) is the internal representation of a
feature. It provides storage for attributes and convenience methods to extract
or mutate them. A feature also contains a dict of ``views``, which are chunks of HTML,
rendered by templates and used to represent a feature in the client.

A record object (`gws.FeatureRecord`) is a data object that only contains a
dict of attributes and, optionally, some metadata properties, depending on the
source. For example, GML feature records usually contain the layer name. It
represents raw source data.

A props object (`gws.FeatureProps`) contains the data necessary to display a feature
in the client. When viewing features, the client only needs their ``uid``, ``shape`` and
``views``. In the edit context, the props object also contains a dict of attributes.
View props (``feature_to_view_props``) only keep the uid and geometry attributes, under
the names ``uid`` and ``geometry``.

Operations
----------

Models are used to perform several abstract operations (`gws.ModelOperation`):

- ``read`` - the client provides a search query (`gws.SearchQuery`) and expects a list of matching props
- ``create`` - the client sends feature props and wants to create new features in the source
- ``update`` - the client sends feature props and wants to update existing features
- ``delete`` - the client sends feature props and wants respective features to be deleted

Before ``create``, the client can also request a new empty feature to be initialized
(``init_feature``) and sent back.

Fields
------

Most models contain a list of field objects (`gws.ModelField`). A field
deals with a subset of feature data, converts it between representations
and validates it.

When a model performs an operation, it delegates it to all its fields in turn.

Fields are either configured explicitly (``fields``), or created automatically from the
source columns (``withAutoFields``, or when no fields are configured), see
`core.Object.configure_auto_fields`.

There are two kinds of fields: scalar fields represent one attribute (column)
in the source itself, and related fields represent features from other models,
linked to the current model.

Values
------

A field can have value objects (`gws.ModelValue`) attached to it.
Value objects provide a ``compute`` method. When a model performs an operation,
and a field has a value object configured for this operation, its ``compute`` method
is called, and the returned value is used as the field value.

Validators
----------

A field can also have validator objects (`gws.ModelValidator`) attached. On
``create`` and ``update``, ``validate_feature`` runs the validators of all fields
and collects a `gws.ModelValidationError` for each field that fails. Each field always has
a ``notEmpty`` and a ``format`` validator: an empty value is an error only for required fields,
and further validators are not run for an empty value. The ``notEmpty`` validator runs first,
then ``format``, then the other validators configured for the operation; validation of a field
stops at the first failure. The error message of a validator defaults to ``validationError_<type>``.

Widgets
-------

A field can have a widget object (`gws.ModelWidget`), which describes how the client
displays and edits the field value. Without a configured widget, related fields of type
``feature`` get a ``featureSelect`` widget, and fields of type ``featurelist`` get a
``featureList`` widget.

Permissions
-----------

To perform a model operation, the user must have the matching permission (`gws.Access`)
on the model: ``read`` to find features, ``create`` to initialize and create features,
``write`` to update and ``delete`` to delete them. Database models raise `gws.ForbiddenError`
when the permission is missing.

Each field can also have permissions, interpreted as follows:

- ``read`` - the content of the field can be read from the source and sent to the client
- ``write`` - user input for this field can be written to the source

It is *not* an error to read or write a field without permission.
The attempt is silently ignored.

If a field has attached value objects, these are applied regardless of field permissions.

Data flow
---------

Database models (`gws.base.database.model`) move data between three representations:
the record (``feature.record.attributes``, raw source values), the feature (``feature.attributes``,
python values) and the props (``feature.props.attributes``, client values).
Scalar fields convert between them with ``raw_to_python``, ``python_to_raw``,
``prop_to_python`` and ``python_to_prop``.

Reading::

    find_features(search, mc)
        for each field: before_select(mc)
            add columns and conditions to mc.dbSelect
        run the select, create a feature with a record for each row
        for each field: after_select(features, mc)
            from_record: record -> feature

Sending to the client::

    feature_to_props(feature, mc)
        for each field: to_props(feature, mc)
            feature -> props, skipped if the user cannot read the field

Writing::

    feature_from_props(props, mc)
        for each field: from_props(feature, mc)
            props -> feature

    create_feature(feature, mc) / update_feature(feature, mc)
        for each field: before_create / before_update
            to_record: feature -> record, skipped for auto and virtual fields
        insert or update the row from the record
        for each field: after_create / after_update

When a scalar field reads a value (``from_record``, ``to_record``), it uses the first value
object configured for the current operation. A value object that is not marked as default
always provides the value. Otherwise the value is taken from the source, if the user
has access to the field, and the default value object is used only if the source has no value.

Context
-------

All model operations require a context data object (`gws.ModelContext`), usually called ``mc``. This object contains:

- the operation (``read``, ``update`` etc.)
- the user performing the operation
- the current project
- the depth of related features to load (``relDepth``, ``maxDepth``)
- other properties, mostly database related

Models provide methods to perform operations, while fields contain callback methods invoked by the model.

For example, here is how a database model implements the ``update`` operation::

    class Model

        def update_feature (feature, mc)

            check if mc.user is allowed to write to this model

            attach an empty record to feature

            open a transaction in the source

            for each field in this model
                invoke the "before_update" callback
                to transfer data from feature.attributes to feature.record

            write changes to the source, using feature.uid as a key and feature.record as data
            (e.g. UPDATE source SET ...record... WHERE id=feature.uid)

            for each field in this model
                invoke the "after_update" callback
                to synchronize updated data, e.g. update a linked model

            commit the transaction

Examples
--------

An editable database model with a value and a validator::

    models+ {
        type "postgres"
        tableName "edit.poi"
        isEditable true
        permissions.edit "allow all"

        fields+ { name "id" type "integer" isPrimaryKey true permissions.edit "deny all" }
        fields+ { name "name" type "text" isRequired true widget { type "input" } }
        fields+ {
            name "updated"
            type "datetime"
            values+ { type "currentTimestamp" forRead false }
        }
        fields+ { name "geom" type "geometry" }
    }

Reading features of a model in Python::

    mc = gws.ModelContext(op=gws.ModelOperation.read, target=gws.ModelReadTarget.map, user=user)
    features = model.find_features(gws.SearchQuery(uids=['1', '2']), mc)
    props = [model.feature_to_props(f, mc) for f in features]
"""

from .core import Config, Object, Props

from . import manager, default_model, util, field, related_field

from .util import (
    iter_features,
    copy_context,
    secondary_context,
)

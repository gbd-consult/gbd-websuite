"""Data models.

A data model, or simply model, is an object that deals with features from
external sources, like database tables, shape files, GML responses, etc.  The
mission of a model is to read features from the source and convert them to a form
suitable for client representation. An "editable" model can also accept features
back from the client, parse, validate and store them.

Features
--------

A feature is a collection of named attributes. One of these attributes can
act as a unique ID (``uid``), and another one can be the feature
geometry (``shape``). ``uid`` is required for editable models, ``shape`` is always optional.

There are three kinds of objects that represent Features:

The feature object (`gws.Feature`) is an internal representation of a
Feature. It provides storage for attributes and convenience methods to extract
or mutate them. A feature also contains a list of ``views``, which are chunks of HTML,
rendered by templates and used to represent a feature in the client.

A record object (`gws.FeatureRecord`) is a data object that only contains a
dict of attributes and, optionally, some metadata properties, depending on the
source. For example, GML feature records usually contain the layer name. It
represents raw source data.

A props object (`gws.FeatureProps`) contains data necessary to display a feature
in the client. When viewing features, the client only needs their ``uid``, ``shape`` and
``views``. In the edit context, the props object also contains a dict of attributes.

Operation modes
---------------

Models are used to perform several abstract operations:

- ``read`` - the client provides a search query (`gws.Search`) and expects a list of matching Props
- ``init`` - the client requests a new empty Feature to be initialized and sent back
- ``create`` - the client sends feature props and wants to create new features in the source
- ``update`` - the client sends feature props and wants to update existing features
- ``delete`` - the client sends feature props and wants respective features to be deleted

Fields
------

Most models contain a collection of field objects (`gws.ModelField`). A field
deals with a subset of feature data and can convert it between representations
and validate it.

When a model performs an operation, it is delegated to all its fields in turn.

A model without fields, called "ad-hoc" or "default" model, can only perform "view" operations.
It simply copies attributes between props and records.

There are two types of fields: "scalar" Fields represent one or multiple
attributes (columns) in the source itself, and "related" fields represent
features from other models, linked to the current model.

Values
------

A field can have "value" (`gws.ModelValue`) objects attached to it.
Value objects provide a ``compute`` method. When a model performs an operation,
and a field has a value object configured for this operation, its ``compute`` method
is called, and the returned value is used as a field's value.

Validators
----------

A field can also have "validator" (`gws.ModelValidator`) objects attached. When
performing ``create`` and ``update`` operations, the model ensures that all
configured validators return ``True``.


Permissions
-----------

To perform a model operation, the user must have a permission to do so.
The permissions (`gws.Access`) are interpreted as follows:

- ``read`` - can perform ``read`` operation
- ``write`` - can perform ``read`` and ``update``
- ``create`` - can perform ``read``, ``init`` and ``create``
- ``delete`` - can perform ``read`` and ``delete``

It is an error to perform a model operation without permission.

Each Field can also have permissions, interpreted as follows:

- ``read`` - the content of the field can be read from the source
- ``write`` - user input for this field can be written to the source

It is *not* an error to attempt to read or write a field without permission.
The attempt is just silently ignored.

Field "read" permissions are applied when a feature is converted to props,
"write" permissions - when it is converted to a record.

If a field has attached value objects, these are applied regardless of field permissions.

Context
-------

All model operation require a context data object (`gws.ModelContext`), usually called ``mc``. This object contains:

- the operation (``read``, ``update`` etc)
- user performing the operation
- current project
- other properties, mostly database related

Models provide methods to perform operations, while fields contain callback methods, invoked by a model.

For example, here's an implementation of the ``update`` operation::


    class Model

        def update_feature (feature, mc)

            check if mc.user is allowed to write to this model

            attach an empty record to feature

            open a transaction in the Source

            for each field in this model
                invoke "before_update" callback
                to transfer data from feature.attributes to feature.record

            write changes to the source, using feature.uid as a key and feature.record as data
            (e.g. UPDATE source SET ...record... WHERE id=feature.uid)

            for each field in this model
                invoke "after_update" callback for each field
                to synchronize updated data, e.g. update a linked model

            commit the transaction

"""

from .core import Config, Object, Props

from . import manager, default_model, util, field, related_field

from .util import (
    iter_features,
    copy_context,
    secondary_context,
)

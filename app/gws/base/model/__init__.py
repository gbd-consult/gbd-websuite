"""Data models.

A data model, or simply Model, is an object that deals with Features
from external sources, like database tables, shape files, gml responses etc.

A mission of a Model is to read Features from a source and convert them to a form
suitable for the client representation. An *editable* Model can, moreover, receive
Features from the client and store them back in a source (not necessarily the same one).

Features
--------

A Feature is a collection of named attributes. One of these attributes can optionally
act as a unique ID (``uid``), and another one can be the feature geometry (``shape``).

There are three kinds of objects that represent Features:

The ``Feature`` object (`gws.core.types.IFeature`) is an internal representation of a Feature.
It provides storage for attributes and convenience methods to extract or mutate them.
A ``Feature`` also contains client ``views``, which are chunks of HTML rendered by templates.

The ``FeatureRecord`` object (`gws.core.types.FeatureRecord`) is a data-only object, that
only contains a dict of attributes and, optionally, some metadata properties,
depending on the source. For example, GML feature records usually contain the layer name.

The ``FeatureProps`` object (`gws.core.types.FeatureProps`) contains data necessary to display
a Feature in the client. When viewing Features, client only needs their ``shape`` and ``views``.
In the edit context, ``FeatureProps`` also contains a dict of attributes.

Operation modes
---------------

Models are used to perform several abstract operations:

- ``view`` - the client provides a SearchQuery (`gws.core.types.SearchQuery`) and expects a list of matching FeatureProps, suitable for viewing
- ``edit`` - the same as ``view``, but the Props are suitable for editing (e.g. contain attributes)
- ``init`` - the client requests a new empty Feature to be initialized and sent back
- ``create`` - the client sends feature props and wants to create new features in the source
- ``update`` - the client sends feature props and wants to update existing features
- ``delete`` - the client sends feature props and wants respective features to be deleted

Fields
------

A Model usually contains a collection of ``Field`` objects (`gws.core.types.IModelField`).
A Field deals with a subset of feature data and can convert it between representations and validate it.

When a Model performs an operation, it is delegated to all its Fields in turn.

A Model without fields, called "ad-hoc" or "default" Model, can only perform ``view`` operations.
It simply copies attributes between Features and FeatureRecords.

There are two types of Fields: "scalar" Fields represent one or multiple attributes (columns) in the Source itself,
"related" Fields represent Features from other Models, linked to the current Model.

Permissions
-----------

To perform a model operation, the user must have a permission to do so.
The permissions (`gws.core.types.Access`) are interpreted as follows:

- ``read`` - can perform ``view`` operation
- ``write`` - can perform ``edit`` and ``update``
- ``create`` - can perform ``edit``, ``init`` and ``create``
- ``delete`` - can perform ``edit`` and ``delete``

It is an error to perform a model operation without permission.

Each Field can also have permissions, interpreted as follows:

- ``read`` - the content of the field can be read from the source
- ``write`` - user input for this field can be written to the source

It is *not* an error to attempt to read or write a field without permission.
The attempt is just silently ignored.

If a Field has fixed or default values defined, these are applied regardless of Field permissions.
Field "read" permissions are applied when a Feature is converted to Props,
"write" permissions - when it's converted to a Record.

Workflows
---------

When an operation is about to be performed, the system creates a ``ModelContext`` data object (``mc``),
which contains:

- operation mode (``view``, ``edit`` etc)
- User performing the operation
- current Project
- other properties, mostly database related

Models provide methods to perform operations, while Fields contain callback methods, which are invoked by a Model.

For example, the ``update`` workflow is designed like this::

    class Model

        def update_features (features)

            for each feature
                attach an empty FeatureRecord to feature

            open a transaction in the Source

            for each field in this model
                invoke "before_update" callback for each field
                this callback is expected to transfer data from feature.attributes to feature.record

            write changes to the source, using feature.uid as a key and feature.record as data
            (e.g. UPDATE source SET ... WHERE id=feature.uid)

            for each field in this model
                invoke "after_update" callback for each field
                this callback is expected to synchronize updated data, e.g. update a linked model

            commit the transaction

"""

from .core import Config, Object, Props
from . import manager, dynamic_model, util
from .default_model import Object as DefaultModel

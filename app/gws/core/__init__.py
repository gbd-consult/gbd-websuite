"""Core library.

The lowest layer of GWS. It provides the basic types, the object tree, logging, constants and
general utilities that every other module depends on. Nothing in ``gws.core`` imports from
other GWS packages.

Most of the package is not imported directly. The ``gws`` package (``gws/__init__.py``) is generated
from ``gws/__init__.pyinc``, which includes the ``*.pyinc`` fragments of this package (and of other
packages) verbatim, and it re-exports the plain modules under short names: ``gws.c``, ``gws.u``,
``gws.log``, ``gws.debug`` and ``gws.env``.

Submodules:

- ``_data.pyinc``: ``gws.Data``, the basic attribute container, which returns ``None`` for missing attributes.
- ``_basic.pyinc``: ``gws.Enum``, common type aliases (``Extent``, ``Point``, ``Duration`` and others),
  the base ``Config`` and ``Props`` classes, command ``Request`` and ``Response`` types,
  ``AttributeType`` and ``GeometryType``.
- ``_access.pyinc``: access modes (``gws.Access``), ACL types and the ``ConfigWithAccess`` base config.
- ``_error.pyinc``: the ``gws.Error`` exception hierarchy.
- ``_tree.pyinc``: the object tree interfaces ``gws.Object``, ``gws.Node`` and ``gws.Root``.
- ``tree_impl``: the implementation of the ``Node`` and ``Root`` methods.
- ``util``: general helpers for data structures, type conversion, files, locks and caching (``gws.u``).
- ``const``: system directories, role names and other constants (``gws.c``).
- ``env``: environment variables that override configuration values (``gws.env``).
- ``log``: a minimal logger writing to stdout (``gws.log``).
- ``debug``: debugging and profiling helpers (``gws.debug``).

Object tree:

All configurable objects are ``gws.Node`` instances in a tree under a ``gws.Root``.
A node is created with ``Root.create`` (or ``Node.create_child``), which looks up the class in the
spec runtime, assigns a uid and calls ``Node.initialize``. Initialization stores the config, computes
the permissions from ``access`` and ``permissions``, and then runs ``pre_configure`` and ``configure``.
These hooks are invoked for every class in the MRO that defines them, base classes first,
so a ``configure`` method does not need to call ``super().configure()``.
If initialization fails, the error is recorded in ``Root.configErrors`` and the node is not created.
After the whole tree is built, ``Root.post_initialize`` runs ``post_configure`` on all nodes,
in the reverse order of creation. ``Root.activate`` calls ``activate`` on all nodes; it is run
after the configuration has been loaded.

The ``.pyinc`` fragments and ``tree_impl`` are linked at import time: ``gws/__init__.py`` sets the
``Access``, ``Error``, ``Data``, ``Props`` and ``Object`` placeholders in ``tree_impl``,
and replaces ``util.is_data_object`` and ``util.to_data_object`` with the real implementations.

Example::

    import gws

    class Object(gws.Node):
        def configure(self):
            self.title = self.cfg('title', default='')
            self.layers = self.create_children(gws.ext.object.layer, self.cfg('layers'))

        def props(self, user):
            return gws.Props(title=self.title, layers=self.layers)

Example::

    import gws

    d = gws.Data(a=1)
    d.b                                    # None
    gws.u.get({'a': {'b': [10, 20]}}, 'a.b.1')  # 20
    gws.u.to_uid('Strasse 1')              # 'strasse_1'

    with gws.u.server_lock('my_task', timeout=5):
        gws.log.info('lock acquired')
"""

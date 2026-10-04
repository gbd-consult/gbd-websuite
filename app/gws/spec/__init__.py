"""Specs: type metadata for configuration, requests and commands.

Specs are metadata that describe the GWS configuration types, request and
response types, extension objects and the command methods of actions. They are
generated from the Python sources (classes, annotations and docstrings) and
used at run time to read and validate configuration and request data, to look
up extension classes and to dispatch commands. The client build and the
documentation generators use the generated specs as well.

The package has two parts: the generator, which creates specs from the
sources, and the runtime, which loads them and works with them.

Modules:

- ``core``: shared data structures: the ``Type`` record, the type kinds ``c``,
  the generator constants ``v``, ``Chunk``, ``SpecData`` and the error classes.
- ``runtime``: creates the ``gws.SpecRuntime`` object (``runtime.create``),
  which generates or loads the specs and provides reading, object and command
  lookups and class loading.
- ``reader``: reads and validates raw values (config dicts, request payloads)
  against spec types. Used by ``runtime.Object.read``.
- ``generator``: the spec generator, see the ``gws.spec.generator`` package.
- ``spec``: command line tool that runs the generator on the developer system
  and writes the specs, the TypeScript API and the configuration references
  to an output directory.
- ``types.pyinc``: the interfaces ``gws.SpecRuntime``,
  ``gws.ApplicationManifest``, ``gws.ExtObjectDescriptor``,
  ``gws.ExtCommandDescriptor``, ``gws.SpecReadOption`` and
  ``gws.CommandCategory``, included into ``gws/__init__.py``.

Design
======

The generator parses the Python sources of the application and its plugins
into a dictionary of ``core.Type`` records keyed by uid. It then resolves
aliases, evaluates defaults, synthesizes variant types for ``gws.ext`` objects
and extracts the types the server needs at run time. The result is a
``core.SpecData`` object, which can be cached as JSON and loaded again
(``generator.main.to_path``, ``generator.main.from_path``).

The runtime wraps a ``SpecData`` object. ``read`` validates a value against a
type using a ``reader.Reader`` and returns the parsed value. ``get_class``
resolves a class reference (a class, a class name or a ``gws.ext`` name) and
imports the defining module on demand. ``command_descriptor`` maps a command
category and name to the action method that handles it. Commands registered
in the ``raw`` category are found under any category. Object and command
descriptors are cached in the runtime object.

Spec Data
=========

``core.SpecData`` is the central data object produced by the generator and
consumed by the runtime. Its fields are:

- ``meta``: build-time metadata: the application version, the manifest path
  and the parsed manifest.
- ``chunks``: source code chunks (the core packages and each plugin) with
  their source files grouped by kind.
- ``serverTypes``: all types the server needs at run time: configuration
  types, request and response types, ext objects and command methods.
- ``strings``: documentation strings keyed by language code (e.g. ``'en'``,
  ``'de'``) and then by type uid.

Types
=====

Each entry in ``serverTypes`` is a ``core.Type`` instance. The ``c`` field
(a ``core.TypeKind`` string) determines the kind of the type and which other
fields are populated. The fields are:

- ``c``: type kind, see below.
- ``uid``: unique identifier, used as the key throughout the spec.
- ``name``: qualified name of named types, e.g. classes and properties.
- ``ident``: source code identifier, used in docs.
- ``constValue``: value of a ``CONSTANT``.
- ``defaultExpression``: unevaluated default (a constant or enum reference),
  evaluated by the normalizer.
- ``defaultValue``: literal default value.
- ``doc``: docstring from the source.
- ``title``: documentation title.
- ``enumDocs``: for ``ENUM``, a ``{member name: docstring}`` dict.
- ``enumValues``: for ``ENUM``, a ``{member name: value}`` dict.
- ``extName``: ``gws.ext`` name, set for extension types and commands.
- ``hasDefault``: ``True`` when a default exists.
- ``isConfig``: ``True`` for types reachable from the application ``Config``.
- ``literalValues``: for ``LITERAL``, the list of allowed values.
- ``modName``, ``modPath``: module that defines the type.
- ``pos``: source position (``path:line``).
- ``tArg``: for ``METHOD``, uid of the last (request) argument.
- ``tArgs``: for ``METHOD``, uids of the arguments in order.
- ``tItem``: for ``LIST`` and ``SET``, uid of the element type.
- ``tItems``: for ``UNION``, ``TUPLE`` and ``CALLABLE``, uids of the member types.
- ``tKey``, ``tValue``: for ``DICT``, uids of the key and value types.
- ``tMembers``: for ``VARIANT``, a ``{tag: uid}`` dict of members.
- ``tModule``: uid of the module type that contains this type.
- ``tOwner``: for ``PROPERTY`` and ``METHOD``, uid of the owning class.
- ``tProperties``: for ``CLASS``, a ``{name: uid}`` dict of properties,
  including inherited ones.
- ``tReturn``: for ``METHOD``, uid of the return type.
- ``tSupers``: for ``CLASS``, uids of the base classes.
- ``tTarget``: for ``TYPE``, ``EXT`` and ``OPTIONAL``, uid of the target type.
- ``tValue``: for ``PROPERTY``, uid of the value type.

Type kinds, defined in ``core.c``:

- ``ATOM``: built-in type: ``any``, ``bool``, ``bytes``, ``float``, ``int``,
  ``str`` and a few other builtins.
- ``CALLABLE``: callable. Uses ``tItems``.
- ``CLASS``: class, e.g. config, props, request and response objects. Uses
  ``tProperties``, ``tSupers``.
- ``CONSTANT``: module-level constant. Uses ``constValue``.
- ``DICT``: ``dict[K, V]``. Uses ``tKey``, ``tValue``.
- ``ENUM``: ``Enum`` subclass. Uses ``enumValues``, ``enumDocs``.
- ``EXT``: a ``gws.ext`` name pointing to a class. Uses ``tTarget``, ``extName``.
- ``LIST``: ``list[T]``. Uses ``tItem``.
- ``LITERAL``: ``Literal[v1, v2, ...]``. Uses ``literalValues``.
- ``METHOD``: a method. Command methods have ``extName`` set to
  ``gws.ext.command.<category>.<name>``. Uses ``tArg``, ``tArgs``,
  ``tReturn``, ``tOwner``.
- ``MODULE``: Python module.
- ``NONE``: the ``None`` type.
- ``OPTIONAL``: ``Optional[T]``. Uses ``tTarget``.
- ``PROPERTY``: a property of a ``CLASS``. Uses ``tOwner``, ``tValue``.
- ``SET``: ``set[T]``. Uses ``tItem``.
- ``TUPLE``: ``tuple[T, ...]``. Uses ``tItems``.
- ``TYPE``: type alias (``TypeAlias``). Uses ``tTarget``.
- ``UNDEFINED``: a type name that could not be resolved.
- ``UNION``: ``Union[T1, T2, ...]`` or ``T1 | T2``. Uses ``tItems``.
- ``VARIANT``: union of the ``gws.ext`` classes of one category, discriminated
  by the ``type`` property. Uses ``tMembers``.

``EXPR`` marks unevaluated default expressions in the generator. ``COMMAND``
and ``FUNCTION`` are declared, but the generator does not produce them.

Example::

    import gws.spec.runtime

    specs = gws.spec.runtime.create('/data/MANIFEST.json', read_cache=True, write_cache=True)

    cfg = specs.read(
        {'type': 'wms', 'provider': {'url': 'https://example.com/wms'}},
        'gws.ext.config.layer',
        path='/data/config.json',
        options={gws.SpecReadOption.verboseErrors},
    )

    cls = specs.get_class('gws.ext.object.layer', 'wms')
    desc = specs.command_descriptor(gws.CommandCategory.api, 'mapGetBox')
"""
"""Specs for the GWS app.

Specs are a set of metadata that describe GWS configuration and runtime objects.
Specs are generated from the source code before the app is run or a build step is performed.

The Specs support module consists of two main components:

- Generator (`gws.spec.generator.main`) that creates Specs from sources
- Runtime (`gws.spec.runtime`) that loads Specs and provides methods to validate configuration or request objects

Generated Specs are also used by the Client builder and documentation generators.

Spec Data
=========

`gws.spec.core.SpecData` is the central data object produced by the Generator and consumed by the Runtime.

It contains the following fields:

- `meta`: Build-time metadata. Includes the application manifest, manifest path, and generator version.

- `chunks`: Source-code "chunks" (collections of related files) that make up the application.

- `serverTypes`: All types the server needs at runtime — configuration types, request/response types, and command descriptors.

- `strings`: Localised documentation strings keyed first by language code (e.g. ``'en'``, ``'de'``) and then by type uid.

Server Types
------------

Each entry in `serverTypes` is a `gws.spec.core.Type` instance. The `c` field (a `gws.spec.core.TypeKind` string)
determines what kind of type it represents and which other fields are populated.

It contains the following fields (not all are populated for every type; see `c`):

- `c`: Type kind (see below).
- `uid`: Unique string identifier, used as key throughout the spec.
- `name`: Short name (e.g. class name, property name).
- `ident`: Fully qualified source-code identifier, used in docs.
- `constValue`: Value for ``CONSTANT`` types.
- `defaultExpression`: Source-expression string for computed defaults.
- `defaultValue`: Literal default value.
- `doc`: Inline docstring from the source.
- `enumDocs`: For ``ENUM`` — ``{member_name → docstring}`` dict.
- `enumValues`: For ``ENUM`` — ``{member_name → value}`` dict.
- `extName`: ``gws.ext`` name, set only for extension types and commands.
- `hasDefault`: ``True`` when a default exists.
- `literalValues`: For ``LITERAL`` — list of allowed literal values.
- `modName` / `modPath`: Module that defines this type.
- `pos`: Source file position (``file:line``).
- `tArg`: For ``METHOD`` — uid of the last (request) argument.
- `tArgs`: For ``METHOD`` — uids of all arguments in order.
- `tItem`: For ``LIST``, ``SET``, ``DICT`` — uid of the element type.
- `tItems`: For ``UNION``, ``TUPLE`` — uids of member types.
- `tKey`: For ``DICT`` — uid of the key type.
- `tMembers`: For ``VARIANT`` — ``{tag → uid}`` map of discriminated members.
- `tModule`: uid of the module type that contains this type.
- `tOwner`: For ``PROPERTY`` — uid of the owning class.
- `tProperties`: For ``CLASS`` — ``{name → uid}`` map of property types.
- `tReturn`: For ``METHOD`` — uid of the return type.
- `tSupers`: For ``CLASS`` — uids of base classes.
- `tTarget`: For ``TYPE``, ``EXT`` — uid of the aliased/target type.
- `tValue`: For ``PROPERTY`` — uid of the property's value type.

The `gws.spec.core.TypeKind` (``c``) field can be one of the following values defined in `gws.spec.core.c`:

- ``ATOM``: Built-in primitive: ``any``, ``bool``, ``bytes``, ``float``, ``int``, ``str``.
- ``CALLABLE``: Untyped callable argument.
- ``CLASS``: User-defined data class (config, props, request, response objects). Uses ``tProperties``, ``tSupers``.
- ``COMMAND``: A ``gws.ext.command.*`` endpoint. Uses ``tArg``, ``tOwner``, ``extName``.
- ``CONSTANT``: Named constant; value stored in ``constValue``.
- ``DICT``: Generic ``dict[K, V]``. Uses ``tKey``, ``tItem``.
- ``ENUM``: Python ``Enum`` subclass. Uses ``enumValues``, ``enumDocs``.
- ``EXPR``: Compile-time expression; not validated at runtime.
- ``EXT``: A ``gws.ext.*`` alias pointing to an extension type. Uses ``tTarget``, ``extName``.
- ``FUNCTION``: Stand-alone callable. Uses ``tArgs``, ``tReturn``.
- ``LIST``: Generic ``list[T]``. Uses ``tItem``.
- ``LITERAL``: ``Literal[v1, v2, …]``. Uses ``literalValues``.
- ``METHOD``: Class method / command handler. Uses ``tArg``, ``tArgs``, ``tReturn``, ``tOwner``.
- ``MODULE``: Python module node; groups types by source file.
- ``NONE``: The ``None`` / ``NoneType`` singleton.
- ``OPTIONAL``: ``Optional[T]`` (i.e. ``T | None``). Uses ``tItem``.
- ``PROPERTY``: A single property slot inside a ``CLASS``. Uses ``tOwner``, ``tValue``.
- ``SET``: Generic ``set[T]``. Uses ``tItem``.
- ``TUPLE``: Generic ``tuple[T, …]``. Uses ``tItems``.
- ``TYPE``: Type alias (``TypeAlias``). Uses ``tTarget``.
- ``UNDEFINED``: Placeholder for a type that could not be resolved.
- ``UNION``: ``Union[T1, T2, …]`` (untagged). Uses ``tItems``.
- ``VARIANT``: Tagged union discriminated by a ``type`` property. Uses ``tMembers``.

"""
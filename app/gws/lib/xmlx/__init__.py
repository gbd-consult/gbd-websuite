"""XML parsing, building and serialization.

An XML document is a tree of ``XmlElement`` objects (``gws.XmlElement``), a subset of the ``ElementTree.Element`` API
with a few extensions. Trees are created by parsing (``from_string``, ``from_path``) or by building (``tag``),
and written with ``XmlElement.to_string``. Options for both directions are in ``gws.XmlOptions``.

Modules
-------

``parser``
    Permissive expat-based parser. Comments and processing instructions are dropped, entity declarations are rejected,
    a DOCTYPE is allowed, undeclared prefixes are accepted, the encoding is detected (UTF-8, then the declared one,
    then Latin-1). Used for remote capabilities and GetFeatureInfo documents (``base/ows/client``, ``plugin/ows_client``),
    OWS POST bodies (``base/ows/server/service``), OGC filters (``base/search/filter``), QGIS projects (``plugin/qgis``),
    SVG icons (``lib/style/icon``) and GekOS records.

``tag``
    ``tag(name, *args, **kwargs)`` builds an element from positional arguments: strings and numbers become text,
    dicts and ``gws.Data`` become attributes, elements become children, iterables are spread, ``None`` is skipped.
    ``name`` can be a slash-separated path (``'a/b/c'`` creates nested elements). This is how OWS server templates
    (``plugin/ows_server/*/templates``, ``base/ows/server/templatelib``), the GML writer (``lib/gml``) and the SVG
    renderer (``lib/svg``) produce XML.

``element``
    The ``XmlElement`` implementation: ElementTree-style navigation (``find``, ``findall``, ``iter``…) plus
    ``textof``, ``textlist``, ``textdict``, ``findfirst``, ``require``, ``isa``, ``add``, ``declare``, ``to_dict``.

``serializer``
    Writes a tree to a string: XML declaration, DOCTYPE, whitespace compaction, escaping, name validation,
    namespace prefixes and declarations. Invoked through ``XmlElement.to_string(opts)``.

``namespace``
    Namespace objects (``gws.XmlNamespace``: ``prefix``, ``uri``, ``schemaLocation``), the table of well-known
    namespaces (``namespace.c.WFS``, ``namespace.c.GML``… from ``namespace_c``), registration of custom namespaces
    (``plugin/xml_helper``), and name utilities (``parse_name``, ``full_name``, ``plain_name``).

``validator``
    Test-only schema validation with lxml against the ``xsi:schemaLocation`` of a document.

``util``
    Value-to-string conversion and escaping.

Names and namespaces
--------------------

Element and attribute names in a tree are either local (``Point``) or Clark names (``{http://www.opengis.net/gml/3.2}Point``).
A prefixed name never occurs in a tree: prefixes are a serialization detail. ``XmlElement.name`` is always the local name.

*Namespace-less handling* (``XmlOptions(removeNamespaces=True)``): the parser strips all prefixes and declarations,
so names are local and paths are plain (``el.find('Capability/Request')``). This is what every OWS reader in the app uses;
the document cannot be reconstructed from the tree.

*Namespace-aware handling* (the default): the parser resolves prefixes against the document's own ``xmlns`` declarations
into Clark names; the default namespace applies to elements, not to attributes; an undeclared prefix ``p`` becomes the
synthetic URI ``adhoc:p``. Declarations are kept in ``XmlElement.namespaces`` of the declaring element, so a parsed
document round-trips through ``to_string`` with its original prefixes (QGIS project patching relies on this).

*Building*: ``tag('OWS_11:Title')`` — a prefix in ``tag()`` names and attribute keys is the *name* of a well-known
namespace (``namespace_c``), resolved to a Clark name at build time; an unknown name raises ``BuildError``.
Elements of configured (custom) namespaces are built with ``namespace.full_name(name, ns)``.

*Serializing*: the prefix of a Clark name comes from the nearest enclosing ``XmlElement.namespaces`` declaration,
else from the namespace table, renamed via ``XmlOptions.customNamespacePrefixes`` (WFS ``NAMESPACES``). An element
in the default namespace (``XmlOptions.defaultNamespace`` or an enclosing ``xmlns=`` declaration) is written unprefixed.
With ``withNamespaceDeclarations``, every namespace used in the tree is declared on the root, with
``xsi:schemaLocation`` if ``withSchemaLocations`` is set; ``XmlElement.declare()`` adds declarations to a specific
element (inline GML for QGIS, ``xlink`` in DTD-based WMS 1.1 output, QName values such as ``gml:Envelope``).

Errors
------

``ParseError`` (malformed input, entity declarations, undecodable bytes), ``BuildError`` (invalid ``tag()`` arguments),
``WriteError`` (invalid or prefixed names in a tree), ``NamespaceError`` (unknown or conflicting namespaces).
All derive from ``xmlx.Error``.
"""

from .parser import from_path, from_string
from .tag import tag
from .error import Error, ParseError, WriteError, NamespaceError, BuildError
from . import namespace, util

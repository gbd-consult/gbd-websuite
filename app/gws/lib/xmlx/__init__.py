"""XML parsing, building and serialization.

An XML document is a tree of ``XmlElement`` objects (the ``gws.XmlElement`` interface), which implement a subset
of the ``ElementTree.Element`` API with a few extensions. Trees are created by parsing (``from_string``, ``from_path``)
or by building (``tag``), and written with ``XmlElement.to_string``. Options for both directions are in ``gws.XmlOptions``.

Modules
-------

``parser``
    Permissive expat-based parser. Comments and processing instructions are dropped, entity declarations are rejected,
    a DOCTYPE is allowed, undeclared prefixes are accepted, the encoding is detected (UTF-8, then the declared one,
    then Latin-1). Used for remote capabilities and GetFeatureInfo documents (``base/ows/client``, ``plugin/ows_client``),
    OWS POST bodies (``base/ows/server/service``), OGC filters (``base/search/filter``), QGIS projects (``plugin/qgis``),
    SVG icons (``lib/style/icon``) and GekOS records.

``tag``
    ``tag(name, *args, **kwargs)`` builds an element from positional arguments: strings, numbers, booleans, dates
    and datetimes become text, dicts and ``gws.Data`` objects become attributes, elements become children,
    other iterables are spread, ``None`` is skipped. Keyword arguments become attributes as well.
    ``name`` can be a slash-separated path (``'a/b/c'`` creates nested elements, the arguments apply to the innermost one);
    slashes inside the ``{uri}`` part of a Clark name do not separate. This is how OWS server templates
    (``plugin/ows_server/*/templates``, ``base/ows/server/templatelib``), the GML writer (``lib/gml``) and the SVG
    renderer (``lib/svg``) produce XML.

``element``
    The ``XmlElement`` implementation: ElementTree-style navigation (``find``, ``findall``, ``iter``...) plus
    ``textof``, ``textlist``, ``textdict``, ``findfirst``, ``require``, ``isa``, ``add``, ``declare``, ``to_dict``.

``serializer``
    Writes a tree to a string: XML declaration, DOCTYPE, whitespace compaction, escaping, name validation,
    namespace prefixes and declarations. Invoked through ``XmlElement.to_string(opts)``.

``namespace``
    Namespace objects (``gws.XmlNamespace``: ``prefix``, ``uri``, ``schemaLocation``), the namespace table,
    registration of custom namespaces (``plugin/xml_helper``), and name utilities (``parse_name``, ``full_name``,
    ``plain_name``).

``namespace_c``
    The well-known namespaces as module-level constants with uppercase names, accessible as ``namespace.c.WFS``,
    ``namespace.c.GML`` and so on.

``validator``
    Test-only schema validation with lxml against the ``xsi:schemaLocation`` of a document. Schemas are downloaded
    and cached under ``gws.c.CACHE_DIR``. Not imported by this package.

``util``
    Value-to-string conversion and escaping.

``error``
    The exception classes.

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

*Building*: in ``tag('OWS_11:Title')``, a prefix in ``tag()`` names and attribute keys is the *name* of a well-known
namespace (``namespace_c``), resolved to a Clark name at build time; an unknown name raises ``BuildError``.
Elements of configured (custom) namespaces are built with ``namespace.full_name(name, ns)``.

*Serializing*: the prefix of a Clark name comes from the nearest enclosing ``XmlElement.namespaces`` declaration,
else from the namespace table, renamed via ``XmlOptions.customNamespacePrefixes`` (WFS ``NAMESPACES``). An element
in the default namespace (``XmlOptions.defaultNamespace`` or an enclosing ``xmlns=`` declaration) is written unprefixed.
With ``withNamespaceDeclarations``, every namespace used in the tree is declared on the root, with
``xsi:schemaLocation`` if ``withSchemaLocations`` is set; ``XmlElement.declare()`` adds declarations to a specific
element (inline GML for QGIS, ``xlink`` in DTD-based WMS 1.1 output, QName values such as ``gml:Envelope``).

*The namespace table* contains the well-known namespaces from ``namespace_c``, keyed by their uppercase name
(``OWS_11``, ``GML``), and custom namespaces registered at configuration time (``namespace.register``), which have no name.
A prefix can occur several times (e.g. ``gml`` for GML 2 and GML 3.2). A URI can occur several times with different
schema locations (``GML_2``, ``GML_3_1``); ``namespace.find_by_uri`` returns the first one, a document that needs
another one declares it explicitly (``XmlElement.namespaces``).

Errors
------

``ParseError`` (malformed input, entity declarations, undecodable bytes), ``BuildError`` (invalid ``tag()`` arguments),
``WriteError`` (invalid or prefixed names in a tree), ``NamespaceError`` (unknown or conflicting namespaces).
All derive from ``xmlx.Error``.

Examples
--------

Read a capabilities document without namespaces::

    import gws.lib.xmlx as xmlx

    root = xmlx.from_string(text, gws.XmlOptions(removeNamespaces=True))
    title = root.textof('Service/Title')
    names = [el.textof('Name') for el in root.findall('Capability/Layer/Layer')]

Parse a document with namespaces, names are Clark names::

    root = xmlx.from_string('<a xmlns:p="u:p"><p:b x="1">t</p:b></a>')
    root.findall('{u:p}b')  # one element
    root.to_string()        # '<a xmlns:p="u:p"><p:b x="1">t</p:b></a>'

Build and serialize an element::

    el = xmlx.tag(
        'geometry/GML:Point',
        {'GML:id': 'xy'},
        xmlx.tag('GML:coordinates', '12.345,56.789'),
        srsName=3857,
    )
    el.to_string(gws.XmlOptions(withNamespaceDeclarations=True))

which returns (line breaks added)::

    <geometry xmlns:gml="http://www.opengis.net/gml/3.2">
        <gml:Point gml:id="xy" srsName="3857">
            <gml:coordinates>12.345,56.789</gml:coordinates>
        </gml:Point>
    </geometry>
"""

from .parser import from_path, from_string
from .tag import tag
from .error import Error, ParseError, WriteError, NamespaceError, BuildError
from . import namespace, util

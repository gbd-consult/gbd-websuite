"""GeoInfoDok schema modules and their generator.

GeoInfoDok is the documentation of the AFIS-ALKIS-ATKIS data model,
published by the AdV (https://www.adv-online.de/GeoInfoDok/). This package
holds Python versions of the schema, generated from the official model
files, which the ALKIS reader and indexer use to type and interpret the
source data.

Submodules:

- ``gid6``: generated schema for GeoInfoDok 6. Each object type, struct,
  union, code list and category of the model is a Python class or type
  alias, with the model documentation as docstrings. Code lists have a
  ``VALUES`` dict of codes and texts. ``METADATA`` describes every type
  (kind, name, GeoInfoDok code, category key, title, attributes and super
  classes) and is used by the reader to choose attribute readers and by the
  indexer to find object types by category. Reading a missing attribute of
  a schema object returns ``None``.
- ``gid7``: the same for GeoInfoDok 7. It is not used by the plugin yet.
- ``generator``: a standalone script that creates ``gid6.py`` and
  ``gid7.py``. For version 6 it parses the Rational Rose ``.cat`` files
  ``Basisschema.cat`` and ``Fachschema.cat``, for version 7 the Enterprise
  Architect (sqlite) file ``AAA-7.1.2.qea``. Only the AAA Basisschema, the
  AFIS-ALKIS-ATKIS Fachschema and the AAA Objektartenkatalog are kept.
- ``extract_props``: a standalone helper script that prints the
  ``GebaeudeProps`` and ``PartProps`` classes and the ``PROPS`` dict for
  ``types``, from the ``gid6`` attributes of the building and Part categories.

The generated modules must not be edited by hand.

Example::

    python3 generator.py 6 /path/to/Basisschema.cat /path/to/Fachschema.cat
    python3 generator.py 7 /path/to/AAA-7.1.2.qea

Example::

    from gws.plugin.alkis.data.geo_info_dok import gid6

    meta = gid6.METADATA['AX_Flurstueck']
    print(meta['kind'], [a['name'] for a in meta['attributes']])
    print(gid6.AX_Bauweise_Gebaeude.VALUES)
"""

"""ALKIS data: reading, indexing, searching and exporting.

This package holds the data layer of the ALKIS plugin. It reads ALKIS source
tables from a PostgreSQL database, builds a compact search index from them in
a separate schema, runs Flurstueck (parcel) and address searches against that
index, and exports search results to files. The ``alkis`` action and the
``gws alkis`` CLI commands are built on top of it.

Submodules:

- ``types``: the data model. ALKIS entities (``Flurstueck``, ``Lage``,
  ``Gebaeude``, ``Buchungsblatt``, ``Person``, ``Part`` and so on), their
  records, ``EnumPair`` code/text values, query objects
  (``FlurstueckQuery``, ``AdresseQuery`` and their options), the
  ``IndexStatus`` and the ``Reader`` interface for source data.
- ``norbit6``: a ``Reader`` for source tables in the GeoInfoDok 6 layout
  written by the norBIT ALKIS import.
- ``indexer``: builds the index. It reads all relevant ALKIS object types
  through a ``Reader``, links them together in memory and writes the result
  into the index tables.
- ``index``: the index node. It defines the index tables, reports the index
  status, finds Flurstuecke and addresses and loads them with their related
  data. It also holds helpers to serialize objects to JSON, to normalize
  search strings and to flatten a Flurstueck into rows for export. The index
  node is created by the ``alkis`` action, which passes the database,
  ``crs``, ``schema``, ``excludeGemarkung`` and ``gemarkungFilter`` options.
  If ``gemarkungFilter`` is set, searches and the street list are restricted
  to these Gemarkung codes.
- ``exporter``: a configurable exporter that writes Flurstuecke to CSV or
  GeoJSON, using models whose field names are flat keys
  (see ``index.flatten_fs``). If no models are configured, a default model
  with basic Flurstueck fields is used.
- ``geo_info_dok``: Python modules generated from the GeoInfoDok schema
  (``gid6``, ``gid7``) and the generator that creates them.

Design:

Every ALKIS object is represented by an *entity* with a list of *records*
(``recs``), one per version in the object's life span, sorted by start date.
A record is historic when its life span has ended; an entity is historic when
all of its records are. Code list values are stored as ``EnumPair`` objects
holding both the code and the display text.

The indexer works in phases: places (Land, Kreis, Gemeinde, Gemarkung and so
on), Buchung (Buchungsblatt, Buchungsstelle, Namensnummer, Person,
Anschrift), Lage with Gebaeude, Flurstueck data, Parts (Nutzung, Festlegung,
Bewertung, computed by intersecting the source geometries with the
Flurstuecke) and finally the flat search tables. Each object is stored as a
JSONB document; the flat ``index*`` tables hold the searchable columns and
refer to Flurstuecke by their uid. Index tables are named
``alkis_<VERSION>_<table>`` and live in the configured index schema.

A search selects matching Flurstueck uids from the flat tables, then loads the
JSONB documents and attaches related data depending on the requested display
themes, dropping historic objects unless history display is requested.

Example::

    from gws.plugin.alkis.data import index, indexer, types as dt

    # build the index from the source schema (ix is an ``index.Object``)
    indexer.run(ix, 'public', with_force=True)

    # find a Flurstueck with its address and buildings
    q = dt.FlurstueckQuery(
        gemarkungCode='1234',
        flurnummer='5',
        zaehler='12',
        options=dt.FlurstueckQueryOptions(
            displayThemes=[dt.DisplayTheme.lage, dt.DisplayTheme.gebaeude],
        ),
    )
    for fs in ix.find_flurstueck(q):
        print(fs.fsnummer, [la.recs[-1].strasse for la in fs.lageList])

Example::

    actions+ {
        type "alkis"
        dbUid "DB_ALKIS"
        dataSchema "public"
        indexSchema "gws8"
        crs 25832
        ui { useExport true }
        exporters+ {
            type "csv"
            title "Basisdaten (CSV)"
            models+ {
                fields [
                    { type "text" name "fs_flurstueckskennzeichen" title "Kennzeichen" }
                    { type "float" name "fs_recs_amtlicheFlaeche" title "Flaeche" }
                ]
            }
        }
    }
"""

from . import index, types

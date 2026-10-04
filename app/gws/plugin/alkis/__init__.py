"""ALKIS cadastre search.

This plugin provides the "Flurstueckssuche": a search for land parcels
(Flurstuecke) and addresses in ALKIS cadastre data stored in PostgreSQL, the
display and printing of parcel details, protected access to land register
(Buchung) and owner (Eigentuemer) data, and the export of parcel data.

The ALKIS source tables (for example, imported by the Norbit plugin) are not
searched directly. A command line indexer reads them once and writes a set of
GWS index tables into a separate schema (``indexSchema``). All searches run
against this index.

Submodules
----------

- ``action`` - the ``alkis`` action. It configures the index, templates,
  printers, exporters and the selection storage, checks access to owner and
  register data, logs owner data access, and implements the API commands
  ``alkisGetToponyms``, ``alkisFindAdresse``, ``alkisFindFlurstueck``,
  ``alkisExportFlurstueck``, ``alkisPrintFlurstueck`` and
  ``alkisSelectionStorage``.
- ``cli`` - the ``alkis`` command line commands: ``index`` (build the index),
  ``status`` (check the index), ``export`` (export all parcels), ``keys``
  (list export keys) and ``dump`` (write the internal representation of
  parcels to a JSON file).
- ``data`` - the data layer:

  - ``data.types`` - records and entities (Flurstueck, Buchungsblatt, Person,
    Lage and others), query and option types.
  - ``data.index`` - the index object: index tables, status, address and
    parcel queries, serialization of entities.
  - ``data.indexer`` - builds the index from the source tables.
  - ``data.norbit6`` - reads source tables in the Norbit (GeoInfoDok 6)
    format.
  - ``data.exporter`` - exports parcels to CSV or GeoJSON.
  - ``data.geo_info_dok`` - the GeoInfoDok 6 and 7 schemas (generated) and
    the generator that creates them.

- ``templates`` - default HTML templates for parcel and address views and
  for printing.
- ``js`` - the client part (search form, results, selection).

Design
------

The action creates one index object per index schema. At activation, the
index status is checked once and cached server-wide; a missing index is
reported as a configuration warning, and if the basic index is missing, the
action is not sent to the client.

Default templates and a default printer are always added after the
configured ones. Without configured exporters, a default exporter is created
when ``ui.useExport`` is set.

Owner and register data are only returned when the index contains them and
the user may read the ``eigentuemer`` and ``buchung`` options respectively.
Requesting owner data implies register data. In control mode, every owner
query requires a control input that matches one of ``controlRules``.

Example::

    actions+ {
        type "alkis"
        dbUid "DB_ALKIS"
        dataSchema "public"
        indexSchema "gws8"
        crs 25832

        ui {
            useSelect true
            useExport true
            searchSpatial true
        }

        eigentuemer {
            permissions.read "allow sachbearbeiter, deny all"
            controlMode true
            controlRules ["^[A-Z]{2}-\\\\d{4}$"]
            logTable "public.alkis_log"
        }

        buchung {
            permissions.read "allow sachbearbeiter, deny all"
        }
    }

Build and check the index from the command line::

    gws alkis index --projectUid myproject
    gws alkis status --projectUid myproject
"""

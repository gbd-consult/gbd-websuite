"""GekoS-Bau integration.

Connects GWS with the GekoS-Bau building permit software
(https://www.gekos.de/). GekoS calls GWS URLs to show positions and parcels on
the map and to get the coordinates of parcels and addresses. Parcels and
addresses are looked up with the ALKIS plugin (``gws.plugin.alkis``).
Optionally, GekoS records (Vorgaenge) are loaded from gek-online into a
PostGIS index table.

Submodules
----------

- ``action`` - the ``gekos`` action. Implements ``gekosGetXY``, which returns
  the coordinates of a parcel or an address as plain text ``x;y``, using the
  ``alkis`` action of the project.
- ``index`` - the index. Loads records from gek-online sources (XML),
  creates point geometries, optionally moves points with the same position
  apart on a circle, and writes the records into the index table. The uid of
  a record is ``<instance>_<ObjectID>``.
- ``core`` - configuration types of the index, its sources and the position
  correction.
- ``cli`` - the command ``gws gekos index``, which creates the index.
- ``js`` - the client part, which handles ``GIS-URL-GetXYFromMap``.

GekoS settings
--------------

The GekoS settings for GWS (Verfahrensadministration / GIS Schnittstelle).
Base address::

    GIS-URL-Base  = http://my-server

Client-side call, handled in the client by the Marker element::

    GIS-URL-ShowXY  = /project/PROJECT_ID/?x=<x>&y=<y>&z=SCALE_VALUE

Client-side call, handled in the client by ``js/index.tsx``::

    GIS-URL-GetXYFromMap = /project/PROJECT_ID/?&x=<x>&y=<y>&gekosUrl=<returl>

Client-side call, handled by the ALKIS plugin::

    GIS-URL-ShowFs = /project/PROJECT_ID/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>

Callback URLs, handled by the ``gekos`` action::

    GIS-URL-GetXYFromFs   = /_/gekosGetXY/projectUid/PROJECT_ID/fs/<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
    GIS-URL-GetXYFromGrd  = /_/gekosGetXY/projectUid/PROJECT_ID/ad/<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

The order of the placeholders must match ``COMBINED_FLURSTUECK_FIELDS`` and
``COMBINED_ADRESSE_FIELDS`` in the ALKIS plugin.

Example::

    actions+ {
        type "gekos"
        index {
            tableName "gekos.vorgaenge"
            crs 25832
            sources [
                {
                    url "https://gek-online.example.com/..."
                    params { ... }
                    instance "bau"
                }
            ]
            position {
                offsetX 0
                offsetY 0
                distance 5
                angle 30
            }
        }
    }

Create the index from the command line::

    gws gekos index --projectUid myproject
"""

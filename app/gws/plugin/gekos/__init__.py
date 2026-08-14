"""Interface with GekoS-Bau software.

See https://www.gekos.de/

GekoS settings for gws (Verfahrensadministration/GIS Schnittstelle)

base address::

    GIS-URL-Base  = http://my-server

client-side call, handled in the client by the Marker element::

    GIS-URL-ShowXY  = /project/PROJECT_ID/?x=<x>&y=<y>&z=SCALE_VALUE

client-side call, handled in the client by js/index.tsx::

    GIS-URL-GetXYFromMap = /project/PROJECT_ID/?&x=<x>&y=<y>&gekosUrl=<returl>

client-side call, handled in the Alkis plugin::

    GIS-URL-ShowFs = /project/PROJECT_ID/?alkisFs=<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>

callback urls, handled by the GekoS action::

    GIS-URL-GetXYFromFs   = /_/gekosGetXY/projectUid/PROJECT_ID/fs/<land>_<gem>_<flur>_<zaehler>_<nenner>_<folge>
    GIS-URL-GetXYFromGrd  = /_/gekosGetXY/projectUid/PROJECT_ID/ad/<str>_<hnr><hnralpha>_<plz>_<ort>_<bishnr><bishnralpha>

NB: the order of placeholders must match ``COMBINED_FLURSTUECK_FIELDS`` and ``COMBINED_ADRESSE_FIELDS`` in the Alkis Plugin

"""

Features
========

GWS enthält Werkzeuge zur Feature Transformation.

Datenmodelle
------------

^REF gws.common.model.Config

Ein *Datenmodel* (``dataModel``) beschreibt wie Attributen eines Features transformiert werden müssen.

Editieren
---------

Vorlagen für visuelle Präsentation
----------------------------------

{TABLE}
    ``feature.title`` | Feature-Titel
    ``feature.teaser`` | Kurzbeschreibung des Features, erscheint in der Autocomplete-Box beim Suchen
    ``feature.description`` | detaillierte Beschreibung, erscheint in der Info-Box
    ``feature.label`` | Kartenbeschriftung für das Feature
{/TABLE}

Die Vorlagen können für Layer (s. ^layer) oder Suchprovider (s. ^search) konfiguriert werden.

Vorlagen für XML Präsentation
-----------------------------

Für WMS/WFS Dienste besteht die Möglichkeit, für bestimmte Features eine angepasste XML Präsentation zu konfigurieren. Dazu erstellen Sie in der Konfiguration der jeweiligen Dienstes eine Vorlage mit dem ``subject`` ``ows.GetFeatureInfo``.

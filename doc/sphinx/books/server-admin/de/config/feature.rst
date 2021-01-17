Features
========

Die GBD WebSuite enthält Werkzeuge mit denen die Features, aus verschiedenen Quellen, einheitlich strukturiert und dargestellt werden können.

Datenmodelle
------------

^REF gws.common.model.Config

Ein *Datenmodel* (``dataModel``) beschreibt wie Attributen eines Quell-Features transformiert werden müssen. Für jedes Attribut eines GBD WebSuite-Features können Sie eine Regel anlegen, mit der Sie für dieses Attribut folgendes definieren:

- eine Bezeichnung (``name``)
- einen Titel (``title``)
- einen Wert. Das kann Quell-Feature Attribut sein (``source``) oder einen festen Wert (``value``) oder eine Formatierungs-String mit ``{...}`` Platzhaltern, die mit Attributen der Quell-Feature ersetzt werden.

Zum Beispiel, wenn eine Postgres Tabelle ``user`` die Spalten ``first_name``, ``last_name`` und ``age`` enthält, können Sie so transformieren: ::

    "dataModel": {
        "rules": [
            { "name": "Name", "format": "{first_name} {last_name}" },
            { "name": "Alter", "source": "age" },
            { "name": "Bezeichnung", "value": "Nutzer" }
        ]
    }

Außerdem können Sie angeben welche Attribute editierbar (``editable``) sind. Wenn Sie eine Editierfunktion verwenden (s. ^digitize und ^tabedit), werden nur editierbare Attribute eines Feature für Editieren freigeschaltet.

Client-Vorlagen
---------------

Sie können Vorlagen (s. ^template) Konfigurieren um die Features an verschiedenen Stellen im Client darzustellen. Die Vorlagen sind mit einem entsprechenden ``subject`` Wert zu versehen

{TABLE head}
    ``subject`` | Funktion
    ``feature.title`` | Feature-Titel
    ``feature.teaser`` | Kurzbeschreibung des Features, erscheint in der Autocomplete-Box beim Suchen
    ``feature.description`` | detaillierte Beschreibung, erscheint in der Info-Box
    ``feature.label`` | Kartenbeschriftung für das Feature
{/TABLE}

Diese Vorlagen können für Layer (s. ^layer) oder Suchprovider (s. ^search) konfiguriert werden.

XML Vorlagen
------------

Für WMS/WFS Dienste besteht die Möglichkeit, für bestimmte Features eine angepasste XML Präsentation zu konfigurieren. Dazu erstellen Sie in der Konfiguration der jeweiligen Dienstes eine Vorlage mit dem ``subject`` ``ows.GetFeatureInfo`` (s. ^ows).

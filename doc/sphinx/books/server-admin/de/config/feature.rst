Features
========

Die GBD WebSuite enthält Werkzeuge mit denen die Features, aus verschiedenen Quellen, einheitlich strukturiert und dargestellt werden können.

Datenmodelle
------------

^REF gws.base.model.Config

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

Seit der Version 7, besteht die Möglichkeit, den Attributen spezielle Editoren bzw Validierungsregel zuzuordnen. Ein Editor kann mit ``editor`` konfiguriert werden: ::

    "dataModel": {
        "rules": [
            {
                "name": "Kommentar",
                "editor": {"type": "text" }
            },
            {
                "name": "Rolle",
                "editor": {"type": "select", "items": [
                    ["admin", "Administrator"],
                    ["user", "Nutzer"],
                    ["guest", "Gast"]
                 ]}
            }
            ...

Es werden folgende Editor-Typen unterstützt:

{TABLE}
   ``string`` | HTML ``<input>`` Element
   ``int`` bzw ``float`` | HTML ``<input type=number>`` Element
    ``text`` | HTML ``<textarea>`` Element
   ``select`` bzw. ``combo`` | ``<select>`` Element, die Werte müssen mit ``items`` als eine Liste von Wert-Titel Paaren konfiguriert werden
   ``checkbox`` | ``<input type=checkbox>`` Element für boolesche Attributen
   ``date`` | Datums-Eingabefeld
{/TABLE}

Wenn kein Editor konfiguriert ist, wird vom Server einen am besten geeigneten Typ gewählt.

Für die Validierungsregel kann eine Liste ``validators`` einem Attribut zugeordnet werden: ::

    "dataModel": {
        "rules": [
            {
                "name": "Kommentar",
                "editor": {"type": "text" },
                "validators": [
                    {"type": "length", "min": 0, "max": 300}
                ]
            },
            {
                "name": "Email",
                "validators": [
                    {"type": "required"},
                    {"type": "regex", "pattern": "^[a-zA-Z0-9.]+@[a-zA-Z.]+$" }
                ]
            },
            ...

Es werden folgende Regel unterstützt:

{TABLE head}
Typ | Parameter |
``required`` |  | der Wert darf nicht leer sein
``range`` | ``min`` und ``max`` | der Wert muss eine Zahl zwischen ``min`` und ``max`` sein
``length`` | ``min`` und ``max`` | die Länge des Stringwerts muss zwischen ``min`` und ``max`` sein
``regex`` | ``pattern`` | der Wert muss mit dem regulären Ausdruck übereinstimmen
{/TABLE}

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

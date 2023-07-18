# Features :/admin-de/config/feature

Die GBD WebSuite enthält Werkzeuge mit denen die Features, aus verschiedenen Quellen, einheitlich strukturiert und dargestellt werden können.

Datenmodelle
------------

%reference_de 'gws.base.model.Config'

Ein *Datenmodel* (`dataModel`) beschreibt wie Attributen eines Quell-Features transformiert werden müssen. Für jedes Attribut eines GBD WebSuite-Features können Sie eine Regel anlegen, mit der Sie für dieses Attribut folgendes definieren:

- eine Bezeichnung (`name`)
- einen Titel (`title`)
- einen Wert. Das kann Quell-Feature Attribut sein (`source`) oder einen festen Wert (`value`) oder eine Formatierungs-String mit `{...}` Platzhaltern, die mit Attributen der Quell-Feature ersetzt werden.

Zum Beispiel, wenn eine Postgres Tabelle `user` die Spalten `first_name`, `last_name` und `age` enthält, können Sie so transformieren:

    "dataModel": {
        "rules": [
            { "name": "Name", "format": "{first_name} {last_name}" },
            { "name": "Alter", "source": "age" },
            { "name": "Bezeichnung", "value": "Nutzer" }
        ]
    }

Außerdem können Sie angeben welche Attribute editierbar (`editable`) sind. Wenn Sie eine Editierfunktion verwenden (s. ^digitize und ^tabedit), werden nur editierbare Attribute eines Feature für Editieren freigeschaltet.

Seit der Version 7, besteht die Möglichkeit, den Attributen spezielle Editoren bzw Validierungsregel zuzuordnen. Ein Editor kann mit `editor` konfiguriert werden:

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

Wenn kein Editor konfiguriert ist, wird vom Server einen am besten geeigneten Typ gewählt.

Für die Validierungsregel kann eine Liste `validators` einem Attribut zugeordnet werden:

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

## Client-Vorlagen

Sie können Vorlagen (siehe [Vorlagen](/admin-de/config/vorlagen)) Konfigurieren um die Features an verschiedenen Stellen im Client darzustellen. Die Vorlagen sind mit einem entsprechenden ``subject`` Wert zu versehen

| ``subject`` | Funktion |
|---|---|
|``feature.title`` | Feature-Titel |
|``feature.teaser`` | Kurzbeschreibung des Features, erscheint in der Autocomplete-Box beim Suchen |
|``feature.description`` | detaillierte Beschreibung, erscheint in der Info-Box |
|``feature.label`` | Kartenbeschriftung für das Feature |

Diese Vorlagen können für Layer (siehe [Layer](/admin-de/config/layer)) oder Suchprovider (siehe [Suche](/admin-de/config/suche)) konfiguriert werden.

## XML Vorlagen

Für WMS/WFS Dienste besteht die Möglichkeit, für bestimmte Features eine angepasste XML Präsentation zu konfigurieren. Dazu erstellen Sie in der Konfiguration der jeweiligen Dienstes eine Vorlage mit dem ``subject`` ``ows.GetFeatureInfo`` (siehe [OWS](/admin-de/config/ows)).
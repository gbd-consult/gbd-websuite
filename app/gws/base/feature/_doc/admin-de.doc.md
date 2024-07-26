# Modelle :/admin-de/config/models

Die GBD WebSuite enthält Werkzeuge mit denen die Features aus verschiedenen Quellen, einheitlich strukturiert und dargestellt werden können.

Datenmodelle
------------

%reference_de 'gws.base.model.Config'

Ein *Datenmodell* (`dataModel`) beschreibt wie Attribute eines Quell-Features transformiert werden müssen. Für jedes Attribut eines GBD WebSuite-Features können Sie eine Regel anlegen, mit der Sie für dieses Attribut folgendes definieren:

- eine Bezeichnung (`name`)
- einen Titel (`title`)
- einen Wert. Das kann ein Quell-Feature Attribut sein (`source`), ein fester Wert (`value`) oder eine Formatierungs-String mit `{...}` Platzhaltern, die mit Attributen der Quell-Feature ersetzt werden.

Zum Beispiel, wenn eine Postgres-Tabelle `user` die Spalten `first_name`, `last_name` und `age` enthält, können Sie diese so transformieren:

    "dataModel": {
        "rules": [
            { "name": "Name", "format": "{first_name} {last_name}" },
            { "name": "Alter", "source": "age" },
            { "name": "Bezeichnung", "value": "Nutzer" }
        ]
    }


Seit der Version 7, besteht die Möglichkeit, den Attributen spezielle Editoren bzw. Validierungsregel zuzuordnen. Ein Editor kann mit `editor` konfiguriert werden:

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

Wenn kein spezifischer Editor konfiguriert ist, wird vom Server ein am besten geeigneter Typ gewählt.

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

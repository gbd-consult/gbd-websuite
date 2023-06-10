# Digitalisierung :/admin-de/plugin/digitalisierung

Die GBD WebSuite enthält eine Digitalisierungsfunktion (``edit``), mit der die Nutzer beliebige Vektor-Objekte, im Client, zeichnen können und diese mit Attributen versehen können. Diese Objekte werden in einer Postgres-Tabelle gespeichert.

Digitalisierung wird freigeschaltet indem Sie bei einem Postgres-Layer (siehe [Datenbanken](/admin-de/config-az/datenbanken)) eine ``edit`` Eigenschaft mit einem Zugriffsblock (siehe [Autorisierung](/admin-de/config-az/autorisierung)) konfigurieren. Den Rollen, die in diesem Zugriffsblock stehen, ist editieren des Layers erlaubt. Im folgenden Beispiel, ist der Layer für alle sichtbar (``all``), jedoch nur von der Rolle ``editor`` editierbar:

```javascript

{
    "type": "postgres",
    "access": [
        { "role": "all", "type": "allow" }
    ],
    ...
    "edit": {
        "access": [
            { "role": "editor", "type": "allow" },
            { "role": "all", "type": "deny" }
        ]
    }
}
```

Die Postgres-Tabelle, die in diesem Layer verwendet wird, muss folgende Kriterien erfüllen:

- die Tabelle muss existieren und eine physische Tabelle sein (d.h. kein View)
- die Tabelle muss eine einzige Geometrie-Spalte enthalten, die einen konkreten Geometrie-Typ besitzt (``POLYGON``, ``POINT`` oder ``LINESTRING``)
- die Tabelle muss einen einspaltigen Primärschlüssel haben

Per default, sind alle Spalten der Tabelle außer Primärschlüssel und Geometrie für Nutzer in einem Web-Formular editierbar. Sie können diese Auswahl mit einem Datenmodell (siehe [Feature](/admin-de/config-az/feature)) anpassen.

Zusätzlich zu der Layer-Konfiguration muss auch die ``edit`` Aktion und das Client Element ``Sidebar.Edit`` aktiviert werden.

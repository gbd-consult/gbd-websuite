# Modell "postgres" :/admin-de/konfiguration/model/postgres

Der Modell-Typ `postgres` liest und schreibt Features aus einer Tabelle einer PostgreSQL/PostGIS-Datenbank. Die Datenquelle bestimmen Sie über `dbUid` (Datenbank-Provider) und `tableName` (Tabelle); mit `sqlFilter` schränken Sie die gelesenen Datensätze zusätzlich ein. Bei entsprechender Konfiguration ist das Modell editierbar.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "postgres"
    tableName "edit.poi"
    isEditable true
    permissions.edit "allow all"
    sort+ { fieldName "name" }

    fields+ {
        name "id"
        type "integer"
        isPrimaryKey true
        permissions.edit "deny all"
    }
    fields+ {
        name "name"
        type "text"
        widget { type "input" }
        textSearch { type "any" }
    }
    fields+ {
        name "geom"
        type "geometry"
    }
}
```

Das Modell liest die Tabelle `edit.poi`; mit `isEditable true` und `permissions.edit "allow all"` wird es editierbar. Über `fields` werden nur die relevanten Spalten aufgeführt: `id` ist der mit `isPrimaryKey true` markierte Schlüssel und wird per `permissions.edit "deny all"` von der Bearbeitung ausgenommen, `name` erhält ein Eingabe-`widget` und wird mit `textSearch` durchsuchbar, `geom` hält die Geometrie. `sort` bestimmt die Vorsortierung. Den Datenbank-Provider geben Sie bei Bedarf über `dbUid` an; ohne Angabe wird der Standard-Provider verwendet.

%ref "gws.plugin.postgres.model.Config"
%demo "postgres_search"

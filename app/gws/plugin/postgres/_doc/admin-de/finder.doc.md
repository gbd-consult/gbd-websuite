# Finder "postgres" :/admin-de/konfiguration/finder/postgres

Der Finder `postgres` durchsucht eine Tabelle einer PostgreSQL/PostGIS-Datenbank und unterstützt Stichwort-, räumliche und Filtersuche. Die Datenquelle bestimmen Sie über `dbUid` (Datenbank-Provider) und `tableName` (Tabelle); mit `sqlFilter` schränken Sie die durchsuchten Datensätze zusätzlich ein.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "postgres"
    tableName "edit.poi"

    models+ {
        type "postgres"
        sort+ { fieldName "name" }

        fields+ {
            name "id"
            type "integer"
            isPrimaryKey true
        }
        fields+ {
            name "name"
            type "text"
            textSearch { type "any" minLength 3 }
        }
        fields+ {
            name "geom"
            type "geometry"
        }
    }

    templates+ {
        type "html"
        subject "feature.title"
        text "{{name}}"
    }
}
```

Der Finder durchsucht die Tabelle `edit.poi`. Das eingebettete `models+` beschreibt die abzufragenden Felder: `name` wird über `textSearch` mit `minLength 3` für die Stichwortsuche freigegeben, `geom` liefert die Geometrie für räumliche Treffer. Über `sort` wird die Ergebnisliste vorsortiert, und das `templates+` mit `subject "feature.title"` bestimmt die Anzeige der Treffer. Den Datenbank-Provider geben Sie bei Bedarf über `dbUid` an.

%ref "gws.plugin.postgres.finder.Config"
%demo "postgres_search"

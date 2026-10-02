# Layer "postgres" :/admin-de/konfiguration/layer/postgres

Ein `postgres`-Layer stellt die Geometrien einer Tabelle einer PostgreSQL/PostGIS-Datenbank als Vektorlayer dar. Die Tabelle geben Sie über `tableName` an; mit `dbUid` wählen Sie bei mehreren konfigurierten Datenbanken den zu verwendenden Provider. Geometrietyp und Koordinatensystem werden aus der Tabelle ermittelt und stehen für Darstellung, Suche und Datenmodelle zur Verfügung.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "POIs"
    type "postgres"
    tableName "edit.poi"
    dbUid "mydb"
    templates+ { subject "feature.label" type "html" text "{{name}}" }
}
```

`tableName` benennt die darzustellende Tabelle im Format `schema.tabelle`; Geometrietyp und Koordinatensystem werden daraus ermittelt. `dbUid` wählt bei mehreren konfigurierten Datenbanken den zu verwendenden Provider – bei nur einer Datenbank entfällt die Angabe. Die `feature.label`-Vorlage beschriftet jedes Feature mit dem Wert des Feldes `name`.

%ref "gws.plugin.postgres.layer.Config"
%demo "postgres_layer"
%demo "postgres_reprojected"
%demo "postgres_search"

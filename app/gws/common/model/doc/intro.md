GWS Modelle beschreiben Konfiguration von strukturen Datenquellen und Formularen. Zu strukturen Datenquellen gehören Datenbanktabellen, Shape- und GeoJSON Daten und WMS/WFS Features.

Ein Model muss eine eindeutige ID haben und besteht aus Feld (`field`) Objekten. Ein Feld Objekt beschreibt eine Spalte, Attribute oder Formularfeld.

Konkretes Beispiel: angenommen wir haben Tabellen `house` (Haus)  und `street` (Strasse) mit dieser Struktur:


@db_diagram
Tabellen "house" und "street".
@db_table "house" "PK id integer|address varchar|floors integer|geom geometry|FK street_id integer"
@db_table "street" "PK id integer|name varchar|geom geometry"
@db_arrow_1 "house:street_id" "street:id"
@end


Für die Tabelle `house` kann das Model ungefähr wie folgt konfiguriert werden:

```
{
    uid "houseModel"

    fields [
        {
            name "id" type "integer" title "ID" isPrimaryKey true
            widget { type "readonly" }
        }
        {
            name "address" type "string" title "Adresse"
            widget { type "input" }
            validators [
                { type "required" message "Adresse darf nicht leer sein"}
            ]
        }
        {
            name "floors" type "integer" title "Etagenzahl"
            widget { type "integer" }
            validators [
                { type "required" message "Bitte Etagenzahl eingeben"}
                { type "range" minValue 1 maxValue 30 message "Etagenzahl muss zwischen 1 und 30 sein"}
            ]
        }
        {
            name "geom" type "geometry" title "Geometrie"
            geometry { type "point" crs 3857 }
            widget { type "geometry" }
            validators [
                { type "geometryWithin" target "POLYGON((0 0) (1 1) (2 2) (0 0))"}
            ]
        }
        {
            name "street" type "relatedFeature" title "Strasse"
            foreignKey { name "street_id" }
        }
    ]
}
```






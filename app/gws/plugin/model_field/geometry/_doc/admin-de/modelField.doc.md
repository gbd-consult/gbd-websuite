# Modell-Feld "geometry" :/admin-de/konfiguration/modelField/geometry

Das Feld `geometry` bildet ein Attribut mit einer Geometrie ab, also der raumbezogenen Gestalt eines Objekts. Mit `geometryType` geben Sie die Art der Geometrie an (etwa Punkt, Linie oder Polygon) und mit `crs` das Koordinatenbezugssystem; werden diese Optionen nicht gesetzt, ermittelt die WebSuite sie aus der Datenbankspalte. Das Feld unterstützt zudem die räumliche Suche.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "geom"
    type "geometry"
    title "Geometrie"
    geometryType "point"
    crs 25832
}
```

Das Feld `geom` bildet die Punktgeometrie eines Objekts ab. Mit `geometryType` und `crs` legen Sie Geometrieart und Koordinatenbezugssystem explizit fest; ohne diese Angaben werden beide aus der Datenbankspalte ermittelt.

%ref "gws.plugin.model_field.geometry.Config"

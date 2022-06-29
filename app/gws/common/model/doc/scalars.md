## Skalare Felder

Skalare Felder repräsentieren Datenbank Spalten und Feature Attribute.

### Primitive Datentypen

Primitive Datentypen entsprechen direkt den Datenbank-Typen. Es werden folgende Datentypen unterstützt:

- `string`
- `integer`
- `float`
- `boolean`
- `date`
- `datetime`

### Komplexe Datentypen

Komplexe Datentypen werden als primitive Datenbank-Typen gespeichert, haben jedoch eine andere Semantik. Zu komplexen Datentypen gehören

- `unit`: numerischer Wert mit einer Maßeinheit, wie `2 m` oder `4 ha`
- `money`: monetärer Wert wie `12.34`
- `currency`: monetärer Wert mit Währung, wie `12.34 EUR`
- `crs` KBS Wert wie `3857`
- `extent`: 4 reelle Zahlen die eine räumliche BoundingBox beschreiben

### Geometrien

Geometrien werden als Feldtyp `geometry` konfiguriert. Einen konkreten Geometrietyp mit KBS beschreibt man mit der Option `geometry`:

```
{
    name "geom"
    type "geometry"
    geometry { type "LineString" crs 3857 }
}
```





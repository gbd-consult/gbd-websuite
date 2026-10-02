# Modell-Widget "geometry" :/admin-de/konfiguration/modelWidget/geometry

Das Widget `geometry` stellt Werkzeuge zum Zeichnen und Bearbeiten der Geometrie eines Objekts bereit und eignet sich für Geometriefelder. Mit `isInline` betten Sie den Editor direkt in das Formular ein, mit `withText` aktivieren Sie zusätzlich die textbasierte Bearbeitung der Geometrie.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "geom"
    type "geometry"
    title "Geometrie"
    widget {
        type "geometry"
        isInline true
        withText true
    }
}
```

Das `geometry`-Feld `geom` erhält Werkzeuge zum Zeichnen und Bearbeiten der Geometrie. Mit `isInline true` wird der Editor direkt in das Formular eingebettet, mit `withText true` steht zusätzlich die textbasierte Bearbeitung der Geometrie zur Verfügung.

%ref "gws.plugin.model_widget.geometry.Config"
%demo "geometry_inline"
%demo "geometry_text"

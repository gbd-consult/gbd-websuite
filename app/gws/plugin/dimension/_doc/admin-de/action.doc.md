# Aktion "dimension" :/admin-de/konfiguration/action/dimension

Die Aktion `dimension` aktiviert das Bemaßungs-Werkzeug im Client. Mit `layerUids` legen Sie die Layer fest, an deren Geometrien gefangen wird, mit `pixelTolerance` die Fangtoleranz in Pixeln und mit `storage` die Ablage für gespeicherte Bemaßungen.

## Darstellung

Eine Bemaßung setzt sich aus mehreren Teilen zusammen, die je eine eigene CSS-Klasse tragen. Über diese Klassen gestalten Sie das Aussehen um; die Regeln legen Sie in einer eigenen [CSS-Datei](/admin-de/konfiguration/style) ab, die zugleich für die Karte und für den Druck gilt.

| Klasse | Bedeutung |
|---|---|
| `.dimensionDimLine` | Maßlinie |
| `.dimensionDimPlumb` | Maßhilfslinie |
| `.dimensionDimExt` | Verlängerung der Maßlinie |
| `.dimensionDimCross` | Kreuz am Messpunkt |
| `.dimensionDimArrow` | Pfeilspitze |
| `.dimensionDimLabel` | Beschriftung |

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "dimension"
    layerUids [ "nrw_kreise_wfs" ]
    pixelTolerance 10
    storage {
        permissions {
            read "allow all"
            write "allow all"
            create "allow all"
        }
    }
}
```

`layerUids` benennt die Layer, an deren Geometrien beim Zeichnen gefangen wird; `pixelTolerance` bestimmt, wie nah der Klick am Stützpunkt liegen muss. `storage` aktiviert das Speichern der Bemaßungen mit eigenen Lese-, Schreib- und Anlege-Rechten.

%ref "gws.plugin.dimension.Config"
%demo "dimension_tool"

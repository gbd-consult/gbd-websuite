# Styling :/admin-de/config/style

Vektor-Feature-Stile können über CSS angepasst werden. Die GBD WebSuite unterstützt standard CSS-Eigenschaften für SVG Symbole (zum Beispiel, `fill`) und einige erweiterte Eigenschaften, welche in Ihrer CSS-Konfiguration mit `--` vorangestellt werden müssen.

| Eigenschaft                        | Standard? | Funktion                                              |
|------------------------------------|-----------|-------------------------------------------------------|
| `fill`                             | ja        | Füllfarbe                                             |
| `stroke`                           | ja        | Kontur `*`                                            |
| `marker`                           | nein      | Markierung für End- bzw Eckpunkte einer Geometrie     |
| `marker-fill`                      | nein      | Füllfarbe der Markierung                              |
| `marker-size`                      | nein      | Größe  der Markierung                                 |
| `marker-stroke`                    | nein      | Kontur der Markierung  `*`                            |
| `label-align`                      | nein      | Beschriftung Ausrichtung                              |
| `label-fill`                       | nein      | Beschriftung Füllfarbe                                |
| `label-font-family`                | nein      | Beschriftung Schrift `**`                             |
| `label-max-scale`                  | nein      | max. Maßstab bei dem die Beschriftungen sichtbar sind |
| `label-min-scale`                  | nein      | min. Maßstab bei dem die Beschriftungen sichtbar sind |
| `label-offset-x`, `label-offset-y` | nein      | X- und Y-Versatz für Beschriftungen                   |
| `label-placement`                  | nein      | Position der Beschriftung (`start`, `end`, `middle`)  |
| `label-stroke`                     | nein      | Kontur der Beschriftung `*`                           |
| `point-size`                       | nein      | Größe für Punkt-Geometrien                            |
| `icon`                             | nein      | eine URL oder Pfad des Icons für Punkt-Geometrien     |

`*` auch `stroke-dasharray`, `stroke-dashoffset`, `stroke-linecap`,  `stroke-linejoin`, `stroke-miterlimit`, `stroke-width`

`**` auch  `label-font-size`, `label-font-style`, `label-font-weight`, `label-line-height`

Styling von Vektorlayern
------------------------

^REF gws.base.style.Config

Für Vektorlayer können Sie Ihre CSS-Regeln direkt in der Konfigurationsdatei unter `style` einfügen:

    "layers": [
        ...
        {
            "title": "My Vector layer",
            "type": "postgres",
            ...
            "style": {
                "type": "css",
                "text": "stroke: rgb(0, 255, 255); stroke-width: 2px; fill: rgba(0, 100, 100, 0.2); --label-fill: rgb(255, 0, 0)"
            }
        }

Alternativ können Sie die Regeln in Ihre CSS-Datei mit aufnehmen und Style-Typ `cssSelector` verwenden

    // In Ihrer CSS-Datei

    .myVectorLayer {
        stroke: rgb(0, 255, 255);
        stroke-width: 2px;
        fill: rgba(0, 100, 100, 0.2);
        --label-fill: rgb(255, 0, 0);
    }

    // In Ihrer config-Datei

    {
        "title": "My Vector layer",
        "type": "postgres",
        ...
        "style": {
            "type": "cssSelector",
            "name": ".myVectorLayer"
        }
    }

^NOTE In der Zukunft, wird auch SLD Styling unterstützt.

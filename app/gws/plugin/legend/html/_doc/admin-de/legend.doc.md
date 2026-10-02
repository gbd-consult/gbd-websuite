# Legende "html" :/admin-de/konfiguration/legend/html

Die Legende `html` erzeugt die Legende aus einer Vorlage. Die Vorlage geben Sie über `template` an; ihr gerendertes Ergebnis wird als Legendenbild verwendet.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Interessante Orte"
    type "postgres"
    tableName "edit.poi"
    legend {
        type "html"
        template {
            type "html"
            path "/data/legends/poi_legend.cx.html"
        }
    }
}
```

Die Legende wird am Layer über den Block `legend` konfiguriert. Der eingebettete `template`-Block ist eine gewöhnliche HTML-Vorlage; ihr gerendertes Ergebnis wird zu einem Bild umgesetzt und als Legende des Layers verwendet. Alternativ zu `path` können Sie den Vorlageninhalt direkt über `text` angeben.

%ref "gws.plugin.legend.html.Config"

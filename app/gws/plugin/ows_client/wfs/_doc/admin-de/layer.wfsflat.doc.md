# Layer "wfsflat" :/admin-de/konfiguration/layer/wfsflat

Ein `wfsflat`-Layer stellt genau einen Quelllayer eines WFS-Dienstes als einzelnen Vektorlayer dar. Den Dienst geben Sie über den `provider` an, mit `sourceLayers` wählen Sie den zu verwendenden Quelllayer aus. Die Features stehen anschließend für Darstellung, Suche und Datenmodelle zur Verfügung.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Kreise und kreisfreie Städte"
    type "wfsflat"
    provider.url "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg"
    sourceLayers.names [ "dvg:nw_dvg1_krs" ]

    templates+ {
        subject "feature.label"
        type "html"
        text "{{gn}}"
    }
}
```

`provider.url` benennt den WFS-Dienst, `sourceLayers.names` wählt daraus genau einen „feature type" über seinen Namen aus. Die Features stehen als Vektorlayer für Darstellung und Suche zur Verfügung. Die `feature.label`-Vorlage beschriftet jedes Feature mit dem Attribut `gn`.

%ref "gws.plugin.ows_client.wfs.flatlayer.Config"
%demo "wfs_flat"

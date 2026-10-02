# Finder "wms" :/admin-de/konfiguration/finder/wms

Der Finder `wms` durchsucht einen WMS-Dienst (Web Map Service) über dessen GetFeatureInfo-Abfrage. Den Dienst geben Sie über den `provider` an; mit `sourceLayers` wählen Sie die abzufragenden Layer aus. Die Suche erfolgt punktbezogen an der angeklickten Position.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "wms"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"
    sourceLayers.names [ "adv_alkis_gewaesser" ]
}
```

Der Finder durchsucht den über `provider.url` angegebenen WMS-Dienst per GetFeatureInfo. Mit `sourceLayers.names` wählen Sie die abzufragenden Layer aus. Die Suche erfolgt punktbezogen an der angeklickten Position.

%ref "gws.plugin.ows_client.wms.finder.Config"

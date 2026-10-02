# Modell "wms" :/admin-de/konfiguration/model/wms

Der Modell-Typ `wms` liest Features aus einem WMS-Dienst (Web Map Service) über dessen GetFeatureInfo-Abfrage. Den Dienst geben Sie über den `provider` an; mit `sourceLayers` wählen Sie die abzufragenden Layer aus. Der Zugriff ist lesend.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "wms"
    provider.url "https://www.wms.nrw.de/geobasis/wms_nw_alkis"
    sourceLayers.names [ "adv_alkis_gewaesser" ]
}
```

Das Modell liest Features aus dem über `provider.url` angegebenen WMS-Dienst per GetFeatureInfo. Mit `sourceLayers.names` wählen Sie die abzufragenden Layer aus. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.ows_client.wms.model.Config"

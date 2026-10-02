# Modell "wfs" :/admin-de/konfiguration/model/wfs

Der Modell-Typ `wfs` liest Features aus einem WFS-Dienst (Web Feature Service). Den Dienst geben Sie über den `provider` an; mit `sourceLayers` wählen Sie aus, welche Layer des Dienstes abgefragt werden. Der Zugriff ist lesend.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "wfs"
    provider.url "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg"
    sourceLayers.names [ "dvg:nw_dvg1_krs" ]
}
```

Das Modell liest Features aus dem über `provider.url` angegebenen WFS-Dienst. Mit `sourceLayers.names` wählen Sie die abzufragenden Feature-Typen aus. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.ows_client.wfs.model.Config"

# Finder "wfs" :/admin-de/konfiguration/finder/wfs

Der Finder `wfs` durchsucht einen WFS-Dienst (Web Feature Service) und unterstützt die räumliche Suche. Den Dienst geben Sie über den `provider` an; mit `sourceLayers` wählen Sie die abzufragenden Layer aus.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "wfs"
    provider.url "https://www.wfs.nrw.de/geobasis/wfs_nw_dvg"
    sourceLayers.names [ "dvg:nw_dvg1_krs" ]
}
```

Der Finder durchsucht den über `provider.url` angegebenen WFS-Dienst. Mit `sourceLayers.names` wählen Sie die abzufragenden Feature-Typen aus. Unterstützt wird die räumliche Suche.

%ref "gws.plugin.ows_client.wfs.finder.Config"

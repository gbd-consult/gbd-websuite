# Finder "qgis" :/admin-de/konfiguration/finder/qgis

Der Finder `qgis` durchsucht ein QGIS-Projekt über den QGIS-Server (GetFeatureInfo). Das Projekt geben Sie über den `provider` an; mit `sourceLayers` wählen Sie die abzufragenden Layer aus. Die Suche erfolgt punktbezogen an der angeklickten Position.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "qgis"
    provider.path "/data/qgis/basiskarte.qgs"
    sourceLayers.names [ "Points of interest" ]
}
```

Der Finder durchsucht das über `provider.path` angegebene QGIS-Projekt. Mit `sourceLayers.names` legen Sie die abzufragenden Layer fest. Die Suche erfolgt punktbezogen an der angeklickten Position über GetFeatureInfo.

%ref "gws.plugin.qgis.finder.Config"

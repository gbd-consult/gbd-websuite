# Modell "qgis" :/admin-de/konfiguration/model/qgis

Der Modell-Typ `qgis` liest Features aus einem QGIS-Projekt über den QGIS-Server (GetFeatureInfo). Das Projekt geben Sie über den `provider` an; mit `sourceLayers` wählen Sie die abzufragenden Layer aus. Der Zugriff ist lesend.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "qgis"
    provider.path "/data/qgis/basiskarte.qgs"
    sourceLayers.names [ "Points of interest" ]
}
```

Das Modell liest Features aus dem über `provider.path` angegebenen QGIS-Projekt. Mit `sourceLayers.names` legen Sie fest, welche Layer des Projekts abgefragt werden; die Attribute werden über GetFeatureInfo bezogen. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.qgis.model.Config"

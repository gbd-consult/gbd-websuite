# OWS-Dienst "wms" :/admin-de/konfiguration/owsService/wms

Der Dienst `wms` stellt die Layer eines Projekts als Web Map Service bereit, sodass externe Anwendungen Kartenbilder abrufen können. Unterstützt werden die WMS-Versionen 1.1.0, 1.1.1 und 1.3.0. Mit `layerLimit` begrenzen Sie die Anzahl der pro Anfrage zulässigen Layer, mit `maxPixelSize` die maximale Breite und Höhe des angeforderten Bildes.

## Vorlagen

Der Dienst unterstützt Vorlagen mit folgenden Subjects:

| Subject | Erzeugtes Dokument |
|---|---|
| `ows.GetCapabilities` | Capabilities-Dokument des Dienstes |
| `ows.GetFeatureInfo` | Sachdaten zu einem abgefragten Objekt |

In der Praxis ist `ows.GetFeatureInfo` die wichtigste: Über sie bestimmen Sie, welche Attribute ein externer Client zu sehen bekommt und in welcher Struktur – etwa um eine INSPIRE-konforme Ausgabe zu erzeugen.

`GetMap` und `GetLegendGraphic` liefern Bilder und lassen sich nicht über Vorlagen anpassen; ihr Ausgabeformat bestimmen die Bildformate des Dienstes.

## Beispiel-Konfiguration ::

```javascript
owsServices+ {
    type "wms"
    supportedCrs [ 4326 3857 25832 ]
    metadata {
        name "MEINE_STADT_WMS"
        title "WMS Meine Stadt"
        abstract "Kartendienst der Stadtverwaltung"
        keywords [ "Stadtplan" "POI" ]
    }
    templates+ {
        subject "ows.GetFeatureInfo"
        type "html"
        path "feature_info.cx.html"
    }
}

map.layers+ {
    title "Points of interest"
    type "qgisflat"
    provider.path "/data/qgis/poi.qgs"
    sourceLayers.names [ "Points of interest" ]
    finders+ { type "postgres" tableName "edit.poi" }
}
```

Der Dienst veröffentlicht die Layer der Projektkarte; über `map.layers` bestimmen Sie also, was der WMS ausliefert. `supportedCrs` legt die Koordinatenbezugssysteme fest, in denen Kartenbilder angefordert werden können. In `metadata` hinterlegen Sie die im Capabilities-Dokument sichtbaren Angaben des Dienstes. Die Vorlage mit dem Subject `ows.GetFeatureInfo` steuert, welche Attribute eine Objektabfrage zurückgibt. Der `finders`-Eintrag des Layers macht diesen abfragbar, sodass `GetFeatureInfo` überhaupt Ergebnisse liefert.

%ref "gws.plugin.ows_server.wms.Config"
%demo "service_wms_advanced_demo"
%demo "service_wms_auth2_demo"
%demo "service_wms_auth_demo"
%demo "service_wms_qgis_demo"
%demo "service_wms_simple_demo"

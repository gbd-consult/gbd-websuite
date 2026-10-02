# Layer "qgis" :/admin-de/konfiguration/layer/qgis

Ein `qgis`-Layer bindet ein QGIS-Projekt als Layerbaum ein und übernimmt dessen Struktur, Darstellung und Legenden. Das Projekt geben Sie über den `provider` an. Mit `compositeRender` lassen Sie den Baum als einzelnes Bild rendern, mit `sqlFilters` hinterlegen Sie SQL-Filter je Quelllayer. Wie beim WMS steuern Sie über `rootLayers`, `excludeLayers` und `flattenLayers` den Aufbau des Baums.

## Das QGIS-Projekt vorbereiten

Die WebSuite übernimmt das Projekt so, wie es ist – Gruppen, Reihenfolge, Symbolisierung, Beschriftungen und Maßstabsgrenzen gelten unverändert. Ein Projekt, das in QGIS richtig aussieht, sieht im Browser gleich aus. Achten Sie beim Vorbereiten auf drei Dinge:

- **Datenquellen müssen auch aus dem Container erreichbar sein.** Ein Projekt, das auf ein lokales Laufwerk des Arbeitsplatzes verweist, funktioniert auf dem Server nicht. Legen Sie Dateidaten unter `/data` ab und binden Sie Datenbanken über einen Dienstnamen aus der [`pg_service.conf`](/admin-de/konfiguration/host) ein – dann gilt dieselbe Verbindung in QGIS und in der WebSuite.
- **Layernamen sind Kennungen.** Über sie sprechen `rootLayers`, `excludeLayers` und `flattenLayers` die Quelllayer an. Werden sie im Projekt umbenannt, greifen diese Angaben nicht mehr.
- **Maßstabsgrenzen aus QGIS werden übernommen.** Ist ein Layer im Browser unsichtbar, obwohl er konfiguriert ist, prüfen Sie zuerst die Maßstabsabhängigkeit im QGIS-Projekt.

## Projekt aus der Datenbank laden

Das Projekt muss nicht als Datei vorliegen. Statt `path` geben Sie `dbUid`, `schema` und `projectName` an; die WebSuite lädt das Projekt dann aus einer PostgreSQL-Datenbank – aus derselben Ablage, in die QGIS beim Speichern in PostgreSQL schreibt.

```javascript
provider {
    dbUid "mydb"
    schema "qgis_projects"
    projectName "basiskarte"
}
```

Das ist besonders dann sinnvoll, wenn mehrere Bearbeiter an derselben Kartengrundlage arbeiten: Ein in QGIS gespeichertes Projekt steht ohne Dateitransfer sofort auf dem Server bereit.

## Änderungen erkennen

Mit `withWatch` überwacht die WebSuite das Projekt und lädt es bei Änderungen neu; `watchFrequency` bestimmt, wie oft geprüft wird. Ohne diese Angabe wird ein geändertes Projekt erst nach einem Neustart wirksam.

## Direkter Zugriff auf die Quellen

Normalerweise rendert und durchsucht der QGIS-Server die Layer. Für einige Quellenarten kann die WebSuite stattdessen unmittelbar auf die Quelle zugreifen und den Umweg über QGIS sparen:

- `directRender` – für Quellen, die selbst Bilder liefern (`wms`, `wmts`, `xyz`),
- `directSearch` – für durchsuchbare Quellen (`wms`, `wfs`, `postgres`).

Beides verbessert die Antwortzeiten. `directSearch` ist zudem die Voraussetzung dafür, dass Werkzeuge wie die Auswahl Geometrien aus einem QGIS-Layer erhalten.

## Beispiel-Konfiguration ::

```javascript
map.layers+ {
    title "Basiskarte"
    type "qgis"
    provider.path "/data/qgis/basiskarte.qgs"
    provider.withWatch true
    provider.directSearch [ "postgres" "wfs" ]
    flattenLayers.level 2
}
```

`provider.path` bindet das QGIS-Projekt als Layerbaum ein; die WebSuite übernimmt dessen Struktur, Symbolisierung und Maßstabsgrenzen unverändert. `provider.withWatch` lädt das Projekt bei Änderungen automatisch neu. `provider.directSearch` lässt durchsuchbare Quellen – hier PostgreSQL und WFS – unmittelbar abfragen und spart den Umweg über den QGIS-Server. `flattenLayers.level 2` fasst die Baumhierarchie ab der zweiten Ebene zu einzelnen Layern zusammen.

%ref "gws.plugin.qgis.layer.Config"
%demo "qgis_auto"
%demo "qgis_bounds"
%demo "qgis_composite"
%demo "qgis_direct_render"
%demo "qgis_layer"

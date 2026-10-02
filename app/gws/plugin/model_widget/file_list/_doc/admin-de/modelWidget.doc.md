# Modell-Widget "fileList" :/admin-de/konfiguration/modelWidget/fileList

Das Widget `fileList` stellt eine Liste angehängter Dateien dar und eignet sich für Beziehungsfelder, die mehrere Dateien aufnehmen. Mit der Option `toFileField` geben Sie das Feld an, über das die Dateien verknüpft werden.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "docs"
    type "relatedFeatureList"
    title "Dokumente"
    fromColumn "id"
    toModel "model_document"
    toColumn "poi_id"
    widget {
        type "fileList"
        toFileField "documentFile"
        withNewButton true
        withUnlinkButton true
    }
}
```

Das Beziehungsfeld `docs` verknüpft mehrere Dokumente aus dem Modell `model_document`. `toFileField` verweist auf das `file`-Feld dieses Modells, über das der Dateiinhalt hochgeladen wird. Mit `withNewButton` und `withUnlinkButton` steuern Sie, ob Schaltflächen zum Hinzufügen und Trennen von Dateien angezeigt werden.

%ref "gws.plugin.model_widget.file_list.Config"
%demo "field_file_list"

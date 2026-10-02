# Modell-Widget "featureList" :/admin-de/konfiguration/modelWidget/featureList

Das Widget `featureList` stellt eine Liste verknüpfter Objekte dar und eignet sich für Beziehungsfelder wie `relatedFeatureList`. Über die Optionen `withNewButton`, `withLinkButton`, `withEditButton`, `withUnlinkButton` und `withDeleteButton` legen Sie fest, welche Schaltflächen zum Anlegen, Verknüpfen, Bearbeiten, Trennen und Löschen von Objekten angezeigt werden.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "pois"
    type "relatedFeatureList"
    title "POIs"
    fromColumn "id"
    toModel "model_poi"
    toColumn "category_id"
    widget {
        type "featureList"
        withNewButton true
        withEditButton true
        withUnlinkButton true
    }
}
```

Das Beziehungsfeld `pois` listet die zugehörigen Objekte aus dem Modell `model_poi` auf, die über `toColumn` mit dem aktuellen Objekt verknüpft sind. Mit `withNewButton`, `withEditButton` und `withUnlinkButton` legen Sie fest, dass Schaltflächen zum Anlegen, Bearbeiten und Trennen von Objekten erscheinen.

%ref "gws.plugin.model_widget.feature_list.Config"
%demo "field_related_linked_feature_list"

# Modell-Feld "relatedFeatureList" :/admin-de/konfiguration/modelField/relatedFeatureList

Das Feld `relatedFeatureList` bildet eine 1:N-Beziehung zu einem untergeordneten Modell ab: Der Wert des Feldes ist die Liste der zugehörigen Kindobjekte. Mit `toModel` benennen Sie das verknüpfte Modell, mit `toColumn` die Fremdschlüsselspalte im Kindmodell und mit `fromColumn` die Schlüsselspalte in dieser Tabelle (standardmäßig deren Primärschlüssel).

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "pois"
    type "relatedFeatureList"
    title "POIs"
    fromColumn "id"
    toModel "model_poi"
    toColumn "category_id"
    widget.type "featureList"
}
```

Das Feld `pois` bildet die 1:M-Gegenrichtung ab: Zu einer Kategorie werden alle POIs geladen, deren `toColumn` (`category_id`) auf deren `fromColumn` (`id`) verweist. Das Widget `featureList` zeigt die zugehörigen Objekte als bearbeitbare Liste.

%ref "gws.plugin.model_field.related_feature_list.Config"
%demo "field_related_feature_list"
%demo "field_related_feature_list_2"

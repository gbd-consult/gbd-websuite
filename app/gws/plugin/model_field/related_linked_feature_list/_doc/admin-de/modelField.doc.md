# Modell-Feld "relatedLinkedFeatureList" :/admin-de/konfiguration/modelField/relatedLinkedFeatureList

Das Feld `relatedLinkedFeatureList` bildet eine N:M-Beziehung zwischen zwei Modellen über eine Zwischentabelle ab: Der Wert des Feldes ist die Liste der verknüpften Objekte. Mit `toModel` benennen Sie das verknüpfte Modell und mit `linkTableName` die Zwischentabelle; `linkFromColumn` und `linkToColumn` bezeichnen darin die Schlüsselspalten für dieses und das verknüpfte Modell.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "tags"
    type "relatedLinkedFeatureList"
    title "Tags"
    toModel "model_tag"
    linkTableName "edit.tagging"
    linkFromColumn "poi_id"
    linkToColumn "tag_id"
    widget {
        type "featureList"
        withUnlinkButton true
    }
}
```

Das Feld `tags` bildet eine M:N-Beziehung über die Verknüpfungstabelle `linkTableName` ab. `linkFromColumn` verweist auf das eigene Modell, `linkToColumn` auf `toModel`. Mit `withUnlinkButton` können bestehende Verknüpfungen im Editor gelöst werden.

%ref "gws.plugin.model_field.related_linked_feature_list.Config"
%demo "field_related_linked_feature_list"

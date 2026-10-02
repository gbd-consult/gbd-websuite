# Modell-Feld "relatedFeature" :/admin-de/konfiguration/modelField/relatedFeature

Das Feld `relatedFeature` bildet eine N:1-Beziehung zu einem übergeordneten Modell ab: Der Wert des Feldes ist das zugehörige Elternobjekt, auf das mehrere Objekte verweisen können. Mit `toModel` benennen Sie das verknüpfte Modell, mit `fromColumn` die Fremdschlüsselspalte in dieser Tabelle und mit `toColumn` die Schlüsselspalte im verknüpften Modell (standardmäßig dessen Primärschlüssel).

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "category"
    type "relatedFeature"
    title "Kategorie"
    fromColumn "category_id"
    toModel "model_category"
    toColumn "id"
    widget.type "featureSelect"
}
```

Das Feld `category` bildet eine M:1-Beziehung ab: Jeder POI verweist über `fromColumn` (`category_id`) auf einen Datensatz im Zielmodell `toModel`, verknüpft über dessen Spalte `toColumn` (`id`). Das Widget `featureSelect` zeigt die verfügbaren Kategorien in einer Auswahlliste.

%ref "gws.plugin.model_field.related_feature.Config"
%demo "field_related_feature"
%demo "field_related_feature_lazy"

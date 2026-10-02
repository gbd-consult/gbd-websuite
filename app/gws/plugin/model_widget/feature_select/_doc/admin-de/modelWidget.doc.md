# Modell-Widget "featureSelect" :/admin-de/konfiguration/modelWidget/featureSelect

Das Widget `featureSelect` stellt verknüpfte Objekte als Auswahlliste dar und eignet sich für Beziehungsfelder, bei denen ein Objekt aus einer festen Menge gewählt wird. Mit der Option `withSearch` blenden Sie ein Suchfeld zum Filtern der Einträge ein.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "category"
    type "relatedFeature"
    title "Kategorie"
    fromColumn "category_id"
    toModel "model_category"
    toColumn "id"
    widget {
        type "featureSelect"
        withSearch true
    }
}
```

Das Beziehungsfeld `category` verweist über `fromColumn` auf ein Objekt des Modells `model_category`. Das Widget stellt die verfügbaren Objekte als Auswahlliste dar; mit `withSearch true` blenden Sie ein Suchfeld zum Filtern der Einträge ein.

%ref "gws.plugin.model_widget.feature_select.Config"

# Modell-Widget "featureSuggest" :/admin-de/konfiguration/modelWidget/featureSuggest

Das Widget `featureSuggest` stellt ein Eingabefeld mit Vorschlagsfunktion dar, das während der Eingabe passende Objekte anbietet. Es eignet sich für Beziehungsfelder, bei denen ein verknüpftes Objekt über eine Autovervollständigung gefunden wird.

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
        type "featureSuggest"
    }
}
```

Das Beziehungsfeld `category` verweist auf ein Objekt des Modells `model_category`. Das Widget `featureSuggest` bietet während der Eingabe passende Objekte per Autovervollständigung an und eignet sich dadurch besonders für Modelle mit vielen Einträgen.

%ref "gws.plugin.model_widget.feature_suggest.Config"

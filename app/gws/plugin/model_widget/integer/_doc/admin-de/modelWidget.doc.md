# Modell-Widget "integer" :/admin-de/konfiguration/modelWidget/integer

Das Widget `integer` stellt ein Eingabefeld für ganze Zahlen dar und eignet sich für Felder vom Typ `integer`. Mit `step` legen Sie die Schrittweite fest, mit `placeholder` einen Platzhaltertext für das leere Feld.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "anzahl"
    type "integer"
    title "Anzahl"
    widget {
        type "integer"
        step 5
        placeholder "0"
    }
}
```

Das `integer`-Feld `anzahl` wird als Eingabefeld für ganze Zahlen dargestellt. Mit `step 5` erhöht oder verringert der Nutzer den Wert über die Schaltflächen in Fünferschritten; `placeholder` bestimmt den Hinweistext im leeren Feld.

%ref "gws.plugin.model_widget.integer.Config"

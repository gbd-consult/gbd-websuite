# Modell-Feld "integer" :/admin-de/konfiguration/modelField/integer

Das Feld `integer` bildet ein Attribut mit einem ganzzahligen Wert ab. Sie verwenden es für numerische Angaben ohne Nachkommastellen, etwa Anzahlen oder Kennziffern.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "anzahl"
    type "integer"
    title "Anzahl"
    widget { type "integer" }
}
```

Das Feld `anzahl` speichert eine ganze Zahl. Das Widget `integer` ist die Vorgabe für `integer`-Felder und lässt nur ganzzahlige Eingaben zu.

%ref "gws.plugin.model_field.integer.Config"

# Modell-Wert "static" :/admin-de/konfiguration/modelValue/static

Die Wertquelle `static` setzt den Feldwert auf einen festen, in der Konfiguration hinterlegten Wert. Sie eignet sich für Konstanten und Vorgabewerte. Den Wert geben Sie über `value` an.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "status"
    type "text"
    title "Status"
    values+ {
        type "static"
        value "neu"
        isDefault true
    }
}
```

Die Wertquelle `static` trägt bei neuen Objekten den festen Wert `neu` in das Feld `status` ein. Mit `isDefault true` gilt der Wert als Vorgabe, die der Bearbeiter noch ändern kann.

%ref "gws.plugin.model_value.static.Config"

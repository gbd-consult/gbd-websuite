# Modell-Wert "format" :/admin-de/konfiguration/modelValue/format

Die Wertquelle `format` bildet den Feldwert aus einer Formatvorlage, in die die Attribute des bearbeiteten Objekts eingesetzt werden. So setzen Sie einen Wert aus mehreren anderen Feldern zusammen. Die Vorlage geben Sie über `format` an, wobei Sie Attribute in geschweiften Klammern referenzieren.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "description"
    type "text"
    title "Beschreibung"
    widget.type "textarea"
    values+ {
        type "format"
        isDefault true
        format "Beschreibung für {{name}} (ID {{id}})"
    }
}
```

Die Wertquelle `format` setzt den Feldwert aus einer Vorlage zusammen, in die Attribute in doppelten geschweiften Klammern eingesetzt werden. Mit `isDefault true` dient der erzeugte Text als Vorschlag, der weiter bearbeitet werden kann.

%ref "gws.plugin.model_value.format.Config"
%demo "value_format"

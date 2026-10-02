# Modell-Validator "notEmpty" :/admin-de/konfiguration/modelValidator/notEmpty

Der Validator `notEmpty` stellt sicher, dass ein Feld einen Wert enthält, und macht es damit zum Pflichtfeld. Leere Zeichenketten und fehlende Werte werden abgelehnt. Automatisch vergebene Felder werden beim Anlegen eines Objekts ausgenommen. Weitere Optionen gibt es nicht.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "name"
    type "text"
    title "Name"
    validators+ {
        type "notEmpty"
    }
}
```

Der Validator `notEmpty` macht das Feld `name` zum Pflichtfeld. Leere Zeichenketten und fehlende Werte werden beim Speichern abgelehnt.

%ref "gws.plugin.model_validator.not_empty.Config"

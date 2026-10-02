# Modell-Validator "dateRange" :/admin-de/konfiguration/modelValidator/dateRange

Der Validator `dateRange` prüft, ob der Wert eines Feldes ein gültiges Datum innerhalb eines vorgegebenen Bereichs ist. Die Grenzen legen Sie über `min` und `max` fest; beide sind selbst Wertquellen und dürfen daher auch berechnet werden. Liegt das Datum außerhalb des Bereichs, wird die Eingabe abgelehnt.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "datum"
    type "date"
    title "Datum"
    validators+ {
        type "dateRange"
        min { type "static" value "2024-01-01" }
        max { type "static" value "2025-01-01" }
    }
}
```

Der Validator `dateRange` beschränkt das Feld `datum` auf den Zeitraum vom 1.1.2024 bis 1.1.2025. Die Grenzen `min` und `max` sind Wertquellen und werden hier über `static` fest gesetzt.

%ref "gws.plugin.model_validator.date_range.Config"
%demo "validators"

# Modell-Wert "currentTimestamp" :/admin-de/konfiguration/modelValue/currentTimestamp

Die Wertquelle `currentTimestamp` setzt den Feldwert auf den aktuellen Zeitstempel zum Zeitpunkt der Bearbeitung. Sie eignet sich für Felder, die den Anlage- oder Änderungszeitpunkt eines Objekts festhalten. Weitere Optionen gibt es nicht.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "updated_at"
    type "date"
    title "Aktualisiert am"
    permissions.edit "deny all"
    values+ {
        type "currentTimestamp"
        forRead false
        forUpdate true
    }
}
```

Die Wertquelle `currentTimestamp` setzt den aktuellen Zeitstempel. Mit `forUpdate true` und `forRead false` wird der Wert bei jeder Speicherung erneuert; `permissions.edit "deny all"` verhindert eine manuelle Eingabe.

%ref "gws.plugin.model_value.current_timestamp.Config"
%demo "value_current_timestamp"

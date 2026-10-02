# Modell-Feld "datetime" :/admin-de/konfiguration/modelField/datetime

Das Feld `datetime` bildet ein Attribut mit einem Zeitpunkt aus Datum und Uhrzeit ab. Sie verwenden es für Zeitstempel; die Werte werden intern und bei der Übertragung im ISO-Format gehalten, die lokale Darstellung übernimmt der Client.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "erfasst_am"
    type "datetime"
    title "Erfasst am"
}
```

Das Feld `erfasst_am` speichert Datum und Uhrzeit als Zeitstempel. Ohne konfiguriertes Widget wird ein einfaches Eingabefeld erzeugt; der Wert wird im ISO-Format gehalten.

%ref "gws.plugin.model_field.datetime.Config"

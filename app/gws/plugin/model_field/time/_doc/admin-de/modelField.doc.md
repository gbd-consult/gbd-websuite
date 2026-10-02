# Modell-Feld "time" :/admin-de/konfiguration/modelField/time

Das Feld `time` bildet ein Attribut mit einer Uhrzeit (ohne Datum) ab. Sie verwenden es für Tageszeit-Angaben; die Werte werden intern und bei der Übertragung im ISO-Format gehalten, die lokale Darstellung übernimmt der Client.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "oeffnung"
    type "time"
    title "Öffnungszeit"
}
```

Das Feld `oeffnung` speichert eine Uhrzeit ohne Datum. Ohne konfiguriertes Widget wird ein einfaches Eingabefeld erzeugt; der Wert wird im ISO-Format gehalten.

%ref "gws.plugin.model_field.time.Config"

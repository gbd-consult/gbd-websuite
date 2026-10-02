# Modell-Feld "date" :/admin-de/konfiguration/modelField/date

Das Feld `date` bildet ein Attribut mit einem Datumswert (ohne Uhrzeit) ab. Sie verwenden es für kalendarische Angaben; die Werte werden intern und bei der Übertragung im ISO-Format gehalten, die lokale Darstellung übernimmt der Client.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "aktualisiert"
    type "date"
    title "Aktualisiert am"
    widget { type "date" }
}
```

Das Feld `aktualisiert` hält ein Datum ohne Uhrzeit. Das Widget `date` zeigt im Editor einen Kalender zur Auswahl; intern wird der Wert im ISO-Format gespeichert.

%ref "gws.plugin.model_field.date.Config"

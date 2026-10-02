# Modell-Widget "date" :/admin-de/konfiguration/modelWidget/date

Das Widget `date` stellt das Attribut als Datumsfeld mit Kalenderauswahl dar. Es eignet sich für Felder vom Typ `date`.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "baubeginn"
    type "date"
    title "Baubeginn"
    widget {
        type "date"
    }
}
```

Das `date`-Feld `baubeginn` wird im Client als Datumsfeld mit Kalenderauswahl dargestellt. Das Widget hat keine weiteren Optionen; es genügt, den Typ `date` anzugeben.

%ref "gws.plugin.model_widget.date.Config"

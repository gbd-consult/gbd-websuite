# Modell-Widget "float" :/admin-de/konfiguration/modelWidget/float

Das Widget `float` stellt ein Eingabefeld für Dezimalzahlen dar und eignet sich für Felder vom Typ `float`. Mit `step` legen Sie die Schrittweite fest, mit `placeholder` einen Platzhaltertext für das leere Feld.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "flaeche"
    type "float"
    title "Fläche"
    widget {
        type "float"
        placeholder "Fläche in m²"
    }
}
```

Das `float`-Feld `flaeche` wird als Eingabefeld für Dezimalzahlen dargestellt. Mit `placeholder` legen Sie den Hinweistext fest, der im leeren Feld erscheint.

%ref "gws.plugin.model_widget.float.Config"

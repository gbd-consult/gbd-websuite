# Modell-Widget "select" :/admin-de/konfiguration/modelWidget/select

Das Widget `select` stellt das Attribut als Auswahlliste dar, aus der genau ein Wert gewählt wird. Die verfügbaren Einträge geben Sie mit `items` als Liste von Wert-Text-Paaren an; mit `withSearch` blenden Sie ein Suchfeld zum Filtern der Einträge ein.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "kategorie"
    type "integer"
    title "Kategorie"
    widget {
        type "select"
        withSearch true
        items [
            { value 1 text "Gastronomie" }
            { value 2 text "Einzelhandel" }
            { value 3 text "Verwaltung" }
        ]
    }
}
```

Das Feld speichert den numerischen Schlüssel der Kategorie, im Client wählt der Nutzer aber den lesbaren Text. Jeder Eintrag in `items` verbindet den gespeicherten `value` mit dem angezeigten `text`. `withSearch true` blendet über der Liste ein Suchfeld ein – nützlich bei langen Auswahllisten.

%ref "gws.plugin.model_widget.select.Config"

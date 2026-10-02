# Modell-Widget "input" :/admin-de/konfiguration/modelWidget/input

Das Widget `input` stellt ein einzeiliges Texteingabefeld dar und eignet sich für Felder vom Typ `text`. Mit der Option `placeholder` legen Sie einen Platzhaltertext für das leere Feld fest.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "name"
    type "text"
    title "Name"
    widget {
        type "input"
        placeholder "Name eingeben"
    }
}
```

Das `text`-Feld `name` wird als einzeiliges Eingabefeld dargestellt. Mit `placeholder` legen Sie den Hinweistext fest, der im leeren Feld angezeigt wird.

%ref "gws.plugin.model_widget.input.Config"

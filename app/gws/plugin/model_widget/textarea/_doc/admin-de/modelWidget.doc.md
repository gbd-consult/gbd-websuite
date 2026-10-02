# Modell-Widget "textarea" :/admin-de/konfiguration/modelWidget/textarea

Das Widget `textarea` stellt ein mehrzeiliges Texteingabefeld dar und eignet sich für Felder mit längeren Textinhalten. Mit `height` legen Sie die Höhe des Feldes fest, mit `placeholder` einen Platzhaltertext für das leere Feld.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "beschreibung"
    type "text"
    title "Beschreibung"
    widget {
        type "textarea"
        height 150
        placeholder "Kommentar eingeben"
    }
}
```

Das `text`-Feld `beschreibung` wird als mehrzeiliges Eingabefeld dargestellt. Mit `height 150` legen Sie die Höhe des Feldes in Pixeln fest; `placeholder` bestimmt den Hinweistext im leeren Feld.

%ref "gws.plugin.model_widget.textarea.Config"

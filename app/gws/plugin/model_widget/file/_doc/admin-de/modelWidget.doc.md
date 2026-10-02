# Modell-Widget "file" :/admin-de/konfiguration/modelWidget/file

Das Widget `file` stellt ein Feld zum Hochladen einer einzelnen Datei dar. Es eignet sich für Feldtypen, die einen Dateianhang aufnehmen.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "dokument"
    type "file"
    title "Dokument"
    contentColumn "content"
    nameColumn "filename"
    widget {
        type "file"
    }
}
```

Das `file`-Feld `dokument` erlaubt das Hochladen einer einzelnen Datei. Der Dateiinhalt wird in der über `contentColumn` angegebenen Spalte gespeichert, der Dateiname in der Spalte aus `nameColumn`. Das Widget selbst hat keine weiteren Optionen.

%ref "gws.plugin.model_widget.file.Config"

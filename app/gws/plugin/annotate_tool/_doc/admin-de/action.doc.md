# Aktion "annotate" :/admin-de/konfiguration/action/annotate

Die Aktion `annotate` aktiviert das Markieren- und Messen-Werkzeug im Client, mit dem Nutzer Geometrien zeichnen und beschriften können. Mit `storage` richten Sie die Ablage für gespeicherte Markierungen ein, mit `labels` legen Sie die Standard-Beschriftungsvorlagen fest.

## Beispiel-Konfiguration ::

```javascript
actions+ {
    type "annotate"
    storage {
        permissions {
            read "allow all"
            write "allow all"
            create "allow all"
        }
    }
}
```

`storage` aktiviert das Speichern und Laden von Markierungen; über die `permissions` legen Sie fest, wer Ablagen lesen (`read`), überschreiben (`write`) und neu anlegen (`create`) darf. Ohne `storage` bleibt das Werkzeug nutzbar, die Markierungen können dann aber nicht dauerhaft gespeichert werden.

%ref "gws.plugin.annotate_tool.action.Config"
%demo "annotate_tool"

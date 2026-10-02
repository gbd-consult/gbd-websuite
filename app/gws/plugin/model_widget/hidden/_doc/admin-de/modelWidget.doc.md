# Modell-Widget "hidden" :/admin-de/konfiguration/modelWidget/hidden

Das Widget `hidden` bindet das Attribut als verstecktes Feld ein, das im Formular nicht sichtbar ist, dessen Wert aber mit dem Objekt gespeichert wird. Es eignet sich für Felder, die technische Werte transportieren, ohne dass der Benutzer sie bearbeitet.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "status"
    type "integer"
    title "Status"
    widget {
        type "hidden"
    }
}
```

Das Feld `status` wird gespeichert, im Formular aber nicht angezeigt. Das Widget `hidden` hat keine weiteren Optionen und eignet sich für technische Werte, die der Benutzer nicht bearbeiten soll.

%ref "gws.plugin.model_widget.hidden.Config"

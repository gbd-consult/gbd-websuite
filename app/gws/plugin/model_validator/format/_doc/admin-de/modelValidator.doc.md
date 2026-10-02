# Modell-Validator "format" :/admin-de/konfiguration/modelValidator/format

Der Validator `format` prüft, ob der Feldwert korrekt eingelesen werden konnte. Kann ein Wert nicht in den Feldtyp umgewandelt werden, etwa eine fehlerhafte Zahl, so lehnt dieser Validator die Eingabe ab, bevor sie gespeichert wird. Weitere Optionen gibt es nicht.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "anzahl"
    type "integer"
    title "Anzahl"
    validators+ {
        type "format"
    }
}
```

Der Validator `format` prüft, ob die Eingabe in den Feldtyp umgewandelt werden konnte. Eine fehlerhafte Zahl im Feld `anzahl` wird so abgelehnt, bevor sie gespeichert wird.

%ref "gws.plugin.model_validator.format.Config"

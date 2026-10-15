# Modell-Validator "format" :/admin-de/konfiguration/modelValidator/format

Der Validator `format` prüft, ob der Feldwert korrekt eingelesen werden konnte. Kann ein Wert nicht in den Feldtyp umgewandelt werden, etwa eine fehlerhafte Zahl, so lehnt dieser Validator die Eingabe ab, bevor sie gespeichert wird. Jedes Feld führt diesen Validator bereits aus; ein konfigurierter `format`-Validator ersetzt den vorgegebenen, etwa um mit `message` eine eigene Fehlermeldung zu setzen.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "anzahl"
    type "integer"
    title "Anzahl"
    validators+ {
        type "format"
        message "Bitte geben Sie eine ganze Zahl ein"
    }
}
```

Der Validator `format` prüft, ob die Eingabe in den Feldtyp umgewandelt werden konnte. Eine fehlerhafte Zahl im Feld `anzahl` wird so abgelehnt, bevor sie gespeichert wird, hier mit einer eigenen Fehlermeldung.

%ref "gws.plugin.model_validator.format.Config"

# Modell-Validator "notEmpty" :/admin-de/konfiguration/modelValidator/notEmpty

Der Validator `notEmpty` prüft, ob ein Feld einen Wert enthält. Leere Zeichenketten, Zeichenketten nur aus Leerzeichen und fehlende Werte gelten als leer. Automatisch vergebene Felder werden beim Anlegen eines Objekts ausgenommen. Jedes Feld führt diesen Validator bereits aus; einen Fehler meldet er aber nur bei Pflichtfeldern (`isRequired`, ohne Angabe aus der Spalte abgeleitet: `NOT NULL`). Ein konfigurierter `notEmpty`-Validator ersetzt den vorgegebenen, etwa um mit `message` eine eigene Fehlermeldung zu setzen.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "name"
    type "text"
    title "Name"
    isRequired true
    validators+ {
        type "notEmpty"
        message "Bitte geben Sie einen Namen ein"
    }
}
```

`isRequired true` macht das Feld `name` zum Pflichtfeld; leere Zeichenketten und fehlende Werte werden beim Speichern abgelehnt. Der Validator `notEmpty` setzt dafür eine eigene Fehlermeldung.

%ref "gws.plugin.model_validator.not_empty.Config"

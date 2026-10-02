# Modell-Validator "regex" :/admin-de/konfiguration/modelValidator/regex

Der Validator `regex` prüft, ob der Wert eines Feldes einem regulären Ausdruck entspricht, und lehnt die Eingabe andernfalls ab. Das Muster geben Sie über `regex` an. Die Prüfung erfolgt mit `re.search`, sodass Sie den Anfangsanker bei Bedarf selbst setzen müssen.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "code"
    type "text"
    title "Code"
    validators+ {
        type "regex"
        regex "(?i)^[0-9][0-9]?[a-z][a-z][a-z]$"
    }
}
```

Der Validator `regex` erlaubt im Feld `code` nur Werte, die dem Muster entsprechen: ein bis zwei Ziffern gefolgt von drei Buchstaben. `(?i)` schaltet die Gross-/Kleinschreibung ab.

%ref "gws.plugin.model_validator.regex.Config"
%demo "validators"

# Modell-Validator "numberRange" :/admin-de/konfiguration/modelValidator/numberRange

Der Validator `numberRange` prüft, ob der Wert eines Feldes eine Zahl innerhalb eines vorgegebenen Bereichs ist. Die Grenzen legen Sie über `min` und `max` fest; beide sind selbst Wertquellen und dürfen daher auch berechnet werden. Liegt die Zahl außerhalb des Bereichs, wird die Eingabe abgelehnt.

## Beispiel-Konfiguration ::

```javascript
fields+ {
    name "anzahl"
    type "integer"
    title "Anzahl"
    validators+ {
        type "numberRange"
        min { type "static" value 100 }
        max { type "static" value 200 }
    }
}
```

Der Validator `numberRange` lässt im Feld `anzahl` nur Werte zwischen 100 und 200 zu. Die Grenzen `min` und `max` sind Wertquellen; hier werden sie über `static` als feste Zahlen angegeben.

%ref "gws.plugin.model_validator.number_range.Config"
%demo "validators"

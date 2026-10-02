# Helfer "csv" :/admin-de/konfiguration/helper/csv

Der Hilfsdienst `csv` erzeugt CSV-Ausgaben, etwa für den Export von Sachdaten. Über `format` legen Sie das Ausgabeformat fest, unter anderem das Feldtrennzeichen `delimiter`, die Textkodierung `encoding` und das Anführungszeichen `quote`.

Sollen die Dateien in einer deutschsprachigen Excel-Installation ohne Nacharbeit korrekt geöffnet werden, haben sich folgende Werte bewährt:

```javascript
format {
    delimiter ";"
    encoding "cp1252"
}
```

Ausschlaggebend ist dabei `delimiter`: Excel erwartet im deutschen Gebietsschema das Semikolon und stellt eine komma-getrennte Datei sonst in einer einzigen Spalte dar.

Das Dezimaltrennzeichen ist **keine** Option des Hilfsdienstes – es ergibt sich aus dem [Gebietsschema](/admin-de/themen/betrieb/lokalisierung). Für ein Komma setzen Sie `locales ["de_DE"]`.

%warn
Die Option `formulaHack` ist **standardmäßig aktiv**: Werte, die nur aus Ziffern bestehen, werden mit einem vorangestellten `=` versehen und in Anführungszeichen gesetzt, damit Excel sie als Formel und nicht als Zahl behandelt. Das bewahrt führende Nullen, verändert aber die Darstellung von Postleitzahlen, Telefonnummern und Kennungen. Setzen Sie `formulaHack false`, wenn die Rohwerte erhalten bleiben sollen.
%end

Mit `quoteAll` setzen Sie jeden Wert in Anführungszeichen, mit `rowDelimiter` das Zeilenende (`CR`, `LF` oder eine feste Zeichenfolge).

## Beispiel-Konfiguration ::

```javascript
helpers+ {
    type "csv"
    format {
        delimiter ";"
        encoding "cp1252"
        quoteAll true
    }
}
```

Dieser Eintrag registriert den CSV-Hilfsdienst für die gesamte Anwendung; sämtliche CSV-Ausgaben verwenden anschließend die hier gesetzten `format`-Werte. Mit `delimiter ";"` und `encoding "cp1252"` sind die Dateien für eine deutschsprachige Excel-Installation vorbereitet, `quoteAll true` setzt jeden Wert in Anführungszeichen. Nicht gesetzte Werte wie `formulaHack` behalten ihre Vorgaben.

%ref "gws.plugin.csv_helper.Config"

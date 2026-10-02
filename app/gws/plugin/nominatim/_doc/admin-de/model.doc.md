# Modell "nominatim" :/admin-de/konfiguration/model/nominatim

Der Modell-Typ `nominatim` liest Features aus dem Nominatim-Dienst von OpenStreetMap und eignet sich für die Adress- und Ortssuche. Über `country` und `language` schränken Sie die Ergebnisse auf ein Land bzw. eine Sprache ein. Das Modell ist ausschließlich lesend.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "nominatim"
    country "de"
    language "de"
}
```

Das Modell bezieht seine Features aus dem Nominatim-Dienst. Mit `country` beschränken Sie die Ergebnisse auf ein Land (ISO-Ländercode) und mit `language` auf eine Ausgabesprache. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.nominatim.model.Config"

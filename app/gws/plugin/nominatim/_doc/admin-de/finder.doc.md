# Finder "nominatim" :/admin-de/konfiguration/finder/nominatim

Der Finder `nominatim` durchsucht den Nominatim-Dienst von OpenStreetMap und eignet sich für die Adress- und Ortssuche. Über `country` und `language` schränken Sie die Ergebnisse auf ein Land bzw. eine Sprache ein.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "nominatim"
    country "de"
    language "de"
    spatialContext "map"
}
```

Der Finder durchsucht den Nominatim-Dienst. Mit `country` beschränken Sie die Ergebnisse auf ein Land und mit `language` auf eine Ausgabesprache. Über `spatialContext "map"` wird die Suche auf die aktuelle Kartenausdehnung bezogen.

%ref "gws.plugin.nominatim.finder.Config"
%demo "nominatim_search"

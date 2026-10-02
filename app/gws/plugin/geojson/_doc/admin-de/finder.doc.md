# Finder "geojson" :/admin-de/konfiguration/finder/geojson

Der Finder `geojson` durchsucht die Features einer GeoJSON-Datei und unterstützt die räumliche Suche. Die Quelle geben Sie über den `provider` und dessen `path` an.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "geojson"
    provider.path "/data/districts.geojson"
}
```

Der Finder durchsucht die Features der über `provider.path` angegebenen GeoJSON-Datei. Unterstützt wird die räumliche Suche innerhalb der angefragten Ausdehnung.

%ref "gws.plugin.geojson.finder.Config"

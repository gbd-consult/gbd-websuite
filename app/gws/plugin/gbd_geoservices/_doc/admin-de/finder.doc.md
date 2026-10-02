# gbd_geoservices :/admin-de/konfiguration/finder/gbd_geoservices

Der Finder `gbd_geoservices` durchsucht die GBD Geoservices von GBD Consult und unterstützt sowohl die Stichwort- als auch die räumliche Suche. Der Zugang erfolgt über den in `apiKey` hinterlegten Schlüssel.

## Beispiel-Konfiguration ::

```javascript
finders+ {
    type "gbd_geoservices"
    apiKey "my_key"
    spatialContext "map"
}
```

Der Finder durchsucht die GBD Geoservices und unterstützt Stichwort- und räumliche Suche. Über `apiKey` hinterlegen Sie den Zugangsschlüssel, mit `spatialContext "map"` wird die Suche auf die aktuelle Kartenausdehnung bezogen.

%ref "gws.plugin.gbd_geoservices.finder.Config"
%demo "geoservices_search"

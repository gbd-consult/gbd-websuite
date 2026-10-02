# gbd_geoservices :/admin-de/konfiguration/model/gbd_geoservices

Der Modell-Typ `gbd_geoservices` liest Features aus den GBD Geoservices von GBD Consult. Der Zugang erfolgt über den in `apiKey` hinterlegten Schlüssel. Das Modell ist ausschließlich lesend; Anlegen, Ändern und Löschen von Features sind nicht möglich.

## Beispiel-Konfiguration ::

```javascript
models+ {
    type "gbd_geoservices"
    apiKey "my_key"
}
```

Das Modell bezieht seine Features aus den GBD Geoservices. Über `apiKey` hinterlegen Sie den für den Zugang erforderlichen Schlüssel. Der Zugriff ist ausschließlich lesend.

%ref "gws.plugin.gbd_geoservices.model.Config"

# wms :/admin-de/config/layer/type/wms

%reference_de 'gws.plugin.ows_provider.wms.wms_layer.Config'

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Falls der Dienst mehrere Layer enthält, werden diese als eine Baumstruktur dargestellt

```javascript

{
    "type": "wms",
    "title": "Webatlas.de - Alle Layer",
    "url": "http://sg.geodatenzentrum.de/wms_dtk250"
}
```

# wmsflat :/admin-de/config/layer/type/wmsflat

%reference_de 'gws.plugin.ows_provider.wfs.wfsflat_layer.Config'

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Die WMS-Layer werden kombiniert, ggf. umprojiziert,  und als einzelnes Bild angezeigt

```javascript

{
    "type": "wmsflat",
    "title": "Webatlas.de - DTK250",
    "sourceLayers": {
        "names": ["dtk250"]
    },
    "url": "http://sg.geodatenzentrum.de/wms_dtk250"
}
```
# wms :/admin-de/config/layer/type/

^REF gws.ext.layer.wms.Config

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Falls der Dienst mehrere Layer enthält, werden diese als eine Baumstruktur dargestellt

    {
        "type": "wms",
        "title": "Webatlas.de - Alle Layer",
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }

# wmsflat :/admin-de/config/layer/type/

^REF gws.ext.layer.wmsflat.Config

Rasterlayer, zeigt Rasterdaten aus einem WMS Dienst. Die WMS-Layer werden kombiniert, ggf. umprojiziert,  und als einzelnes Bild angezeigt

    {
        "type": "wmsflat",
        "title": "Webatlas.de - DTK250",
        "sourceLayers": {
            "names": ["dtk250"]
        },
        "url": "http://sg.geodatenzentrum.de/wms_dtk250"
    }


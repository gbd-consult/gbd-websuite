OWS Services
============

GBD WebSuite can act as an OWS (OGC Web Services) server. You can freely configure these services for each project and each layer separately.


Supported services
------------------


Currently these services are supported:

wms
~~~

WMS is fully implemented according to the version 1.3.0 specification.

wfs
~~~

WFS is implemented according to the version 2.0 specification. The WFS service provides the following capabilities:

- `ImplementsBasicWFS`
- `KVPEncoding`
- `ImplementsResultPaging`

Application and project configuration
-------------------------------------

To enable OWS services for a whole application or a project, add an ``ows`` action to the application/project configuration action block. In the action configuration, specify which services are enabled. Additionally, you can provide an xml namespace name and url for ``GetFeature`` documents. ::

    {
        "type": "ows",
        "services": [
            {
                "type": "wms",
                "enabled": true
            },
            {
                "type": "wfs",
                "enabled": true
            }
        ],
        "xmlNamespace": "mynamespace",
        "xmlNamespaceUri": "https://my-site.com/mynamespace"
    }


Layer configuration
-------------------

Most of the time, the suitable service type is inferred from the layer type, e.g. an image layer is available for the WMS service, a vector layer for a WFS service. You can also specify the services in the layer configuration, and optionally provide a value for the layer OWS name ::

    {
        "type": "qgis",
        "title": "My qgis project",
        "owsName": "qgis_for_wms",
        "ows": {
            "services": [
                {
                    "type": "wms",
                    "enabled": true
                },
                {
                    "type": "wfs",
                    "enabled": false
                }
            ]
        }
    }

External service layers are automatically "cascaded".

URL rewriting
-------------

By default, OWS services are exposed under a dynamic URL, for example ::

    https://my-server.com/_/cmd=owsHttpGet&projectUid=my_project

If you want to rewrite this url to a nicer form (see :doc:`web` for details), for example ::

    https://my-server.com/services/my_project

you have to also provide a ``reversedRewrite`` rule in the ``web`` configuration for WMS/WFS capabilities documents ::

    "reversedRewrite": [
        {
            "match": "^cmd=owsHttpGet&projectUid=([a-z0-9_-]+)$",
            "target": "/services/$1?"
        }
    ]






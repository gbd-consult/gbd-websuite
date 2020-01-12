OWS Dienste
===========

Die GBD WebSuite kann als OWS (OGC Web Services) Server fungieren. Sie können diese Dienste für jedes Projekt frei konfigurieren.


Unterstützte Dienste
--------------------


Derzeit werden folgende Dienste unterstützt:

wms
~~~

Der WMS-Dienst sind vollständig gemäß der Eigenschaften der Version 1.3.0 implementiert.

wfs
~~~

Der WFS-Dienst sind vollständig gemäß der Eigenschaften der Version 2.0 implementiert. Der WFS-Dienst bietet die folgenden Funktionen:

- `ImplementsBasicWFS`
- `KVPEncoding`
- `ImplementsResultPaging`

inspirewms
~~~

inspirewfs
~~~

meta
~~~

ISO19115 metadata service


csw
~~~



Anwendungs- und Projektkonfiguration
----------------------------------

Um OWS-Dienste für eine gesamte Anwendung oder ein Projekt zu aktivieren, müssen Sie eine `` ows`` Aktion  für den Aktionsblock der Anwendungs- / Projektkonfiguration hinzufügen. Geben Sie in der Aktionskonfiguration an, welche Dienste aktiviert sind. Darüber hinaus können Sie einen XML-Namespace-Namen und eine URL für ``GetFeature`` Dokumente angeben. ::

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


Layerkonfiguration
------------------

Meistens wird der geeignete Diensttyp vom Layertyp abgeleitet, z.B. steht für den WMS-Dienst ein Rasterlayer zur Verfügung, für den WFS-Dienst ein Vektorlayer. Sie können die Dienste auch in der Layerkonfiguration konfigurieren und optional einen Wert für den OWS-Namen des Layers angeben:

    {
        "type": "qgis",
        "title": "My qgis project",
        "ows": {
            "name": "myFeature",
            "servicesEnabled": ["wms", "wfs"]
        }
    }

Externe Service-Layer werden automatisch "kaskadiert".

Umschreiben der URL/URL rewriting
---------------------------------

Standardmäßig werden OWS-Dienste unter einer dynamischen URL angezeigt, zum Beispiel ::

    https://my-server.com/_/cmd=owsHttpGet&projectUid=my_project

Wenn Sie diese URL in eine schönere Form umschreiben möchten (siehe: doc: `web` für Details), zum Beispiel:

    https://my-server.com/services/my_project

Sie müssen außerdem eine  ``reversedRewrite`` Regel in der `` web`` Konfiguration für WMS / WFS Capabilities Dokumenten angeben ::

    "reversedRewrite": [
        {
            "match": "^cmd=owsHttpGet&projectUid=([a-z0-9_-]+)$",
            "target": "/services/$1?"
        }
    ]

# Metadaten :/admin-de/config-az/metadata

TODO! %reference_de 'gws.lib.metadata.Config'

GBD WebSuite Metadata konfiguration ist umfangreich und beinhaltet folgende Themen:

- allgemeine Metadaten, wie z.B. Titel (``title``), Beschreibung (``abstract``) und Kontakt-Daten (``contact``)
- INSPIRE Metadaten, wie z.B. Thema (``inspireTheme``) oder Ressourcen-Typ (``inspireResourceType``)
- ISO 19139 Metadaten, wie Scope (``isoScope``) oder Präsentation-Typ (``isoSpatialRepresentationType``)

Die Metadaten können bei Projekten, Layern und OWS Diensten separat konfiguriert werden. Die Layer-Metadaten werden bei der Abbildung vom Layer im einem OWS Dienst verwendet. Auch im Client werden die Metadaten benutzt, z.B. um die Attribution-Leiste zu zeigen. Die Dienst- und Projekt-Metadaten werden zum Erstellen vom Capabilities Dokumenten verwendet, sodass die im Dienst-Eigenschaften konfigurierten Daten von Projekt-Metadaten teilweise überschrieben und präzisiert werden können. Zum Beispiel, wenn Sie in der App-Konfig folgendes haben:

```javascript

    "actions": [
        ...
        {
            "type": "ows",
            "services": [
                "type": "wms",
                "uid": "my_wms_service",
                "meta": {
                    "name": "WMS",
                    "title": "Mein WMS Dienst",
                }
            }
        }
        ...
```

und in einer Projekt-Konfiguration folgendes:

```javascript

    ...
    "uid": "meinprojekt",
    "meta": {
        "abstract": "Mehr über Mein Projekt"
        }
    }
    ...
```

dann werden die Capabilities unter der URL

    http://example.com/_?cmd=owsHttpService&uid=my_wms_service&projectUid=meinprojekt&request=GetCapabilities

wie folgt abgebildet:

```xml
    ...
    <Service>
        <Name>WMS</Name>
        <Title>Mein WMS Dienst</Title>
        <Abstract>Mehr über Mein Projekt</Abstract>
    ...
```

Wenn Sie einen CSW Dienst konfigurieren, sind die Metadaten von allen Objekten unter CSW ``GetRecordById`` Anfrage in ISO bzw Dublin Core Format als separate Dokumente abrufbar (siehe [OWS](/admin-de/plugin/ows)).

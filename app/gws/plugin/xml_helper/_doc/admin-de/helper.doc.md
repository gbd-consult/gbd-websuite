# Helfer "xml" :/admin-de/konfiguration/helper/xml

Der Hilfsdienst `xml` stellt zusätzliche XML-Namensräume für die XML-Erzeugung bereit, wie sie etwa von OWS-Diensten benötigt werden. Über `namespaces` definieren Sie eigene Namensräume mit Präfix, URI, Schema-Ort und Version.

## Beispiel-Konfiguration ::

```javascript
helpers+ {
    type "xml"
    namespaces+ {
        xmlns "demo"
        uri "https://example.com/ows/namespace/demo"
        schemaLocation "https://example.com/ows/namespace/demo.xsd"
    }
}
```

Dieser Eintrag definiert den Namensraum mit dem Präfix `demo`. `uri` ist die Namensraum-URI, `schemaLocation` verweist auf das zugehörige XML-Schema. Auf dieses Präfix greifen Sie anschließend in Layer- und Dienst-Konfigurationen zu, etwa mit `ows.xmlns "demo"` bei WFS-Layern, damit die ausgegebenen Objektarten in diesem Namensraum erscheinen.

%ref "gws.plugin.xml_helper.Config"

Caching-Framework
======================


Der gebündelte Mapproxy-Server kann so konfiguriert werden, dass er Geodaten aus externen WMS-, WMTS- oder KVP-Quellen zwischenspeichert. In der GBD WebSuite verwenden Sie die Optionen ``Cache`` und ``Grid`` in der Map-Konfiguration oder in der Layer-Basis-Konfiguration. Die Konfiguration ``Seeding`` muss in der Hauptkonfiguration verwendet werden.

Es sind einige Schritte notwendig, um einen Layern zwischenspeichern zu können:

* der Layer muss einen definierten ``View`` mit einem ``Extent`` und einem Satz erlaubter ``resolutions`` oder ``scales`` haben. Diese Einstellungen können in der Layer-Konfiguration definiert oder von der Map-Konfiguration vererbt werden.

* der Layer (oder die Karte) muss ein ``Grid`` haben. Für WMS-Quellen ist es wichtig, das Meta-Tiling richtig einzustellen, um das Problem der "baumelnden Labels" zu vermeiden (siehe https://mapproxy. org/docs/latest/labeling. html).

* der Layer (oder die Karte) muss einen ``Cache`` mit ``enabled`` auf ``true`` gesetzt haben

Sobald das Caching eingerichtet ist, wird es automatisch gefüllt, wenn Benutzer ihre Karten durchsuchen. Sie können den Cache auch mit den Kommandozeilen-Tools ``gws cache`` voreinstellen.

Wichtig: Wenn Sie Ansichts- oder Rasterkonfigurationen ändern, müssen Sie den Cache für die Ebene oder die Karte entfernen, um unangenehme Artefakte zu vermeiden.

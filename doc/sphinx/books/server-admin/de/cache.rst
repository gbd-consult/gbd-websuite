Caching-Framework
======================


Der gebündelte Mapproxy-Server kann so konfiguriert werden, dass er Geodaten aus externen WMS-, WMTS- oder KVP-Quellen zwischenspeichert. In der GBD WebSuite verwenden Sie die Optionen ``Cache`` und ``Grid`` in der Map-Konfiguration oder auf Layer-Basis und die Konfiguration ``Seeding`` in der Hauptkonfiguration.

Es sind einige Schritte notwendig, um eine Ebene zwischenzuspeichern:

* die Ebene muss einen definierten ``View`` mit einem ``Extent`` und einem Satz erlaubter ``resolutions`` oder ``scales`` haben. Diese Einstellungen können in der Layer-Konfiguration definiert oder von der Map-Konfiguration vererbt werden.

* die Ebene (oder die Karte) muss ein ``Grid`` haben. Für WMS-Quellen ist es wichtig, das Meta-Tiling richtig einzustellen, um das Problem der "baumelnden Labels" zu vermeiden (siehe https://mapproxy. org/docs/latest/labeling. html).

* die Ebene (oder die Karte) muss einen ``Cache`` mit ``freigegeben`` auf ``true`` gesetzt haben

Sobald das Caching eingerichtet ist, wird es automatisch gefüllt, wenn Benutzer Ihre Karten durchsuchen. Sie können den Cache auch mit den Kommandozeilen-Tools ``gws cache`` voreinstellen.

Wichtig: Wenn Sie Ansichts- oder Rasterkonfigurationen ändern, müssen Sie den Cache für die Ebene oder die Karte entfernen, um unangenehme Artefakte zu vermeiden. 


# Caching-Framework :/admin-de/config/cache

Der GBD WebSuite Server kann Geo-Bilder aus externen Quellen auf der Festplatte zwischenspeichern (bzw. *cachen*), sodass weitere Anfragen zu der selben Quelle viel schneller ausgeführt werden können.  Das Cache Verzeichnis befindet sich in dem von Ihnen konfigurierten *var* Verzeichnis und kann bei Bedarf jederzeit komplett gelöscht werden

%info
 Die Caches können sehr viel Speicherplatz benötigen. Sorgen Sie dafür, dass Ihr Dateisystem über ausreichend freien Platz und freie **inodes** verfügt.
%end

## Layer Konfiguration

Das Cachen kann für jeden Layer mit den Optionen ``cache`` und ``grid`` flexibel konfiguriert werden.

### cache

%reference_de 'gws.base.layer.core.CacheConfig'

Geben Sie hier an, ob Caching aktiviert ist und für wie lange die gecachten Bilder gespeichert werden sollen.

### grid

%reference_de 'gws.base.layer.core.GridConfig'

Geben Sie hier an, wie der Kachelgrid für diesen Layer aussieht. Bei den Layern die keine Kachel liefern, wie ``wms`` oder ``qgisflat`` ist es wichtig einen ausreichenen Puffer (``reqBuffer``) zu setzen, damit die Beschriftungen richtig positioniert werden.

### Seeding

TODO! %reference_de 'gws.base.application.SeedingConfig'

TODO! ^CLIREF cache.seed

Sobald der Cache eingerichtet ist, wird er automatisch gefüllt wenn Benutzer Ihre Karten in Browser anschauen. Sie können den Cache auch mit den Kommandozeilen-Tools ``gws cache`` befüllen (sogenanntes *Seeding*).

## Verwaltung von Cache

TODO! ^CLIREF cache.clean

Mit dem selben Tool können Sie den Status des Cache abfragen oder individuelle Caches löschen.

%info
 Wenn Sie Ansichts- oder Rasterkonfigurationen ändern, müssen Sie den Cache für die Ebene oder die Karte entfernen, um unangenehme Artefakte zu vermeiden.
%end

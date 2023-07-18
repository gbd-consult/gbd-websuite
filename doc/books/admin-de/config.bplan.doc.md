# Bauleitplanung :/admin-de/config/bplan

TODO! %reference_de 'gws.ext.action.bplan.Config'

Diese Aktion unterstützt die Administrations-Oberfläche für Hochladen und Verwalten von Bauleitplanungen. Die Bauleitpläne können in Form eines Zip-Archives hochgeladen werden, die die Shape-Files sowie georeferenzierte Rasterbilder (im `png/pgw` Format) und Dokumente im `pdf` Format enthalten. Der GBD WebSuite Server konvertiert diese Archive in QGIS Projekte, die Geltungsbereiche als Postgres Layer und Rasterbilder als `vrt` Layer enthalten. Die Sachdaten und Geometrien aus Shape-Files werden in einer Postgres Tabelle gespeichert. Zusätzlich besteht es die Möglichkeit, die Metadaten für eine Bauplan-Sammlung (z.B. einer Gemeinde) zu editieren und in einer Postgres Tabelle abzuspeichern.

In der Konfiguration der Aktion können Sie die Daten- und Meta-Tabellen konfigurieren, sowie das Verzeichnis in dem die Bilder und Dokumente gespeichert werden.

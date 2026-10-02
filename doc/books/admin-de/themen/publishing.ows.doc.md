# OWS-Dienste :/admin-de/themen/publishing/ows

Die GBD WebSuite kann ihre Daten nicht nur im eigenen Client zeigen, sondern auch über standardisierte OGC-Webdienste bereitstellen, sodass externe Anwendungen sie nutzen können. Unterstützt werden Darstellungsdienste für Rasterdaten wie WMS und WMTS, Objektdienste für Vektordaten wie WFS sowie ein Katalogdienst (CSW), über den sich die Metadaten abfragen lassen.

Dienste lassen sich global oder je Projekt konfigurieren. Ein global konfigurierter Dienst steht allen Projekten zur Verfügung; ein im Projekt konfigurierter Dienst gilt nur für dieses Projekt. Aus den Layern des jeweiligen Projekts wird ein Wurzel-Layer bestimmt, dessen Baum dem Dienst zugrunde liegt; für jeden Layer lässt sich festlegen, ob und in welchen Diensten er erscheint. Wie beim übrigen Zugriff greifen auch hier die Berechtigungen, sodass sich Dienste auf bestimmte Rollen einschränken lassen.

Standardmäßig sind die Dienste über eine technische Adresse mit Dienst- und Projektkennung erreichbar; über Rewrite-Regeln geben Sie ihnen lesbare Adressen. Die notwendigen XML-Dokumente erzeugt der Server selbst, sie lassen sich jedoch über Vorlagen anpassen.

%see
Siehe auch: [Konfiguration/OWS-Dienste](/admin-de/konfiguration/owsService), [Aktion/ows](/admin-de/konfiguration/action/ows).
%end

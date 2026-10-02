# Drucken :/admin-de/themen/darstellung/drucken

Drucken bedeutet in der GBD WebSuite, ein PDF vom Server zu erhalten. Dazu gibt es zwei Wege.

Zum einen kann der Server ein PDF aus einer Druckvorlage erzeugen. Die Vorlage bestimmt Aufbau und Inhalt der Ausgabe – Seitenformat, Ränder, feste Texte sowie die Stellen, an denen Karte, Legende, Kopf- und Fußzeile eingesetzt werden. Eine Vorlage kann sich über mehrere Seiten erstrecken; dabei stehen die aktuelle Seitenzahl und die Gesamtzahl der Seiten zur Verfügung, etwa für eine Seitennummerierung.

Zum anderen kann der Server ein PDF unmittelbar aus der aktuellen Kartenansicht erzeugen. Es enthält dann alle sichtbaren Layer samt ihren Legenden, ohne dass eine eigene Vorlage nötig ist.

Ein Projekt kann mehrere Druckvorlagen bereitstellen, zwischen denen der Nutzer im Client wählt. Für jede Vorlage lassen sich Qualitätsstufen festlegen, die Geschwindigkeit und Auflösung gegeneinander abwägen.

## Bildschirmfoto

Vom Drucken zu unterscheiden ist das Bildschirmfoto (Client-Element `Toolbar.Screenshot`). Es liefert kein PDF, sondern ein **PNG** eines frei wählbaren Kartenausschnitts: Der Nutzer zieht einen Rahmen auf und gibt die gewünschte Auflösung (DPI) vor, woraus sich die Pixelmaße des Bildes ergeben. Erzeugt wird das Bild vom selben Drucker-Backend wie der Druck, es benötigt also ebenfalls eine aktive Aktion `printer`.

%see
Siehe auch: [Konfiguration/Drucker](/admin-de/konfiguration/printer), [Aktion/printer](/admin-de/konfiguration/action/printer).
%end

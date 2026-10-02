# Layer :/admin-de/themen/karten/layer

Ein Layer stellt eine Ebene der Karte dar. In seiner Konfiguration legen Sie einen Titel fest, der im Ebenenbaum erscheint, bestimmen das Erscheinungsbild und geben an, aus welcher Quelle das Kartenmaterial stammt. Ein Layer kann ein eigenes Ausmaß und eigene Zoomstufen definieren; liegt der Kartenmaßstab außerhalb dieses Bereichs, wird der Layer nicht angezeigt.

Grundsätzlich gibt es zwei Arten von Layern. Ein Rasterlayer liefert Geoinformation als fertiges Pixelbild und enthält selbst keine Sachdaten, lässt sich aber mit einer Suche verbinden. Ein Vektorlayer besteht aus einzelnen Objekten (Features) mit Geometrie und Attributen; sein Erscheinungsbild wird über Stilregeln festgelegt.

Innerhalb der Karte wird ein Layer über eine vollständige Kennung angesprochen, die Projekt, Karte und Layer umfasst. Diese Form nutzen Sie überall dort, wo an anderer Stelle auf einen Layer verwiesen wird.

## Rasterlayer

Für Rasterlayer bestimmen Sie unter anderem das Bildformat, in dem die Kacheln geliefert werden, den Anzeigemodus – ob der Server den ganzen Ausschnitt, einzelne Kacheln oder eine clientseitige Darstellung erzeugt – sowie das Zwischenspeichern der Bilder auf dem Server.

## Vektorlayer

Für Vektorlayer bestimmen Sie das Datenmodell, das Aussehen der Features und ob und wie sie editiert werden dürfen. Über die Ladestrategie legen Sie fest, ob alle Features auf einmal oder nur die des sichtbaren Ausschnitts geladen werden.

Das Aussehen beschreiben Sie mit CSS: Füllung und Linien der Geometrie, Marker an den Stützpunkten sowie Schrift und Lage der Beschriftung. Die WebSuite versteht dafür einen Teil der Standard-CSS-Eigenschaften und ergänzt sie um eigene. Die Regeln geben Sie entweder unmittelbar in der Konfiguration an oder verweisen auf einen Selektor aus einer eigenen CSS-Datei; über den Maßstabsbereich einer Beschriftung steuern Sie zudem, ab wann sie erscheint. Die vollständige Liste der [CSS-Eigenschaften](/admin-de/konfiguration/style) gehört zur Konfigurationsreferenz.

%see
Siehe auch: [Konfiguration/Layer](/admin-de/konfiguration/layer), [Konfiguration/Modelle](/admin-de/konfiguration/model), [Konfiguration/CSS-Stile](/admin-de/konfiguration/style).
%end

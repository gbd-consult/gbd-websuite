# Client :/admin-de/themen/darstellung/client

Der Client ist die JavaScript-Anwendung, mit der Nutzer die Karte im Browser bedienen. Sein Aufbau wird über ein Client-Objekt gesteuert, das es global und je Projekt gibt; projektbezogene Angaben überschreiben die globalen.

Die Oberfläche besteht aus einer Werkzeugleiste, einer Seitenleiste und einer Infoleiste am unteren Rand. Über eine geordnete Liste von Elementen bestimmen Sie, welche Bausteine erscheinen und in welcher Reihenfolge – etwa der Ebenenbaum, die Suche oder die Anzeige des Maßstabs. Jedes Element lässt sich auf bestimmte Rollen beschränken, sodass Werkzeuge nur berechtigten Nutzern angezeigt werden. Allgemeine Optionen steuern das grundsätzliche Verhalten, etwa ob die Seiten- oder Infoleiste anfangs sichtbar ist.

## Werkzeuge

Die meisten Werkzeuge der Werkzeugleiste erklären sich von selbst; ein Werkzeug erscheint nur, wenn sein Client-Element aufgeführt ist – und bei serverseitigen Funktionen zusätzlich die zugehörige Aktion aktiv ist. Einige Werkzeuge haben ein nicht offensichtliches Verhalten:

- **Objektabfrage** (*Identify*): Es gibt zwei Varianten. `Toolbar.Identify.Click` fragt beim Klick auf die Karte ab – und bei gedrückter Umschalttaste zusätzlich bei Mausbewegung. `Toolbar.Identify.Hover` fragt fortlaufend an der Mausposition ab.
- **Standort** (`Toolbar.Location`): nutzt die Standortermittlung des Browsers und setzt daher HTTPS sowie die Zustimmung des Nutzers voraus. Der ermittelte Standort wird mit einem Genauigkeitskreis dargestellt und angesteuert; liegt er außerhalb des Kartenausschnitts, erscheint eine Meldung.
- **Lupe** (`Toolbar.Lens`): Der Nutzer zeichnet eine Geometrie – Punkt, Linie, Rechteck, Polygon oder Kreis – und kann sie anschließend verschieben und verformen. Die Geometrie treibt fortlaufend eine räumliche Suche.
- **Bildschirmfoto** (`Toolbar.Screenshot`): erzeugt ein PNG des gewählten Ausschnitts, siehe [](/admin-de/themen/darstellung/drucken).

## Layer-Flags

Unabhängig von den Oberflächenelementen trägt jeder Layer eine Reihe von Schaltern, die dem Client mitteilen, wie er im Ebenenbaum erscheinen soll – ob er anfangs sichtbar oder verborgen ist, im Baum aufgeklappt wird oder ganz ausgeblendet bleibt. So passen Sie die Darstellung des Ebenenbaums an, ohne die Layer selbst zu verändern.

%see
Siehe auch: [Konfiguration/Client](/admin-de/konfiguration/client).
%end

%ref "gws.base.client.core.Config"

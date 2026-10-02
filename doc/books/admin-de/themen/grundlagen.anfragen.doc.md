# Verarbeitung von Web-Anfragen :/admin-de/themen/grundlagen/anfragen

Die GBD WebSuite beantwortet zwei Arten von Anfragen.

Statische Dateien wie HTML-Seiten, Bilder oder PDFs liefert sie wie ein gewöhnlicher Webserver aus. Der eigentliche Zweck sind jedoch dynamische Inhalte: Kartenbilder, Suchergebnisse und Sachdaten, die der Server je nach Anfrage erzeugt.

Alle dynamischen Anfragen laufen über einen einzigen Endpunkt, erkennbar am Unterstrich `_` im Pfad. Das erste Pfadsegment danach benennt den Befehl, gefolgt von Parameter-Wert-Paaren:

    http://example.com/_/mapHttpGetBox/projectUid/london/layerUid/london.map.metro

Welcher Befehl eine Anfrage bearbeitet, entscheidet, welche *Server-Aktion* zuständig ist. Eine Aktion ist eine Funktionsgruppe des Servers – etwa für Karten, Suche oder Druck – und liefert je nach Anfrage HTML, JSON oder ein Bild zurück. Aktionen müssen in der Konfiguration freigeschaltet werden; ist eine Aktion nicht aktiv, steht die zugehörige Funktion nicht zur Verfügung.

Diese technische Form der Adressen muss nach außen nicht sichtbar sein: Über die [Rewrite-Regeln einer Webseite](/admin-de/themen/grundlagen/andere-inhalte) bilden Sie lesbare URLs auf interne Aufrufe ab.

Bevor eine Anfrage bearbeitet wird, stellt der Server fest, welcher Nutzer sie stellt, und prüft dessen [Berechtigung](/admin-de/themen/zugriff) für das angesprochene Objekt; ohne Anmeldung gilt der Nutzer als anonym.

%see
Siehe auch: [Konfiguration/Aktionen](/admin-de/konfiguration/action).
%end

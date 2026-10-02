# Projekte :/admin-de/themen/grundlagen/architektur/projekte

Ein Projekt ist die zentrale Anwendungseinheit der GBD WebSuite. Es bündelt eine Karte mit ihren Layern, die Suche, Druckvorlagen und weitere Funktionen zu einer eigenständigen Anwendung. Auf einer Installation können beliebig viele Projekte parallel laufen.

Jedes Projekt hat eine eindeutige Kennung, die in der Adresse der Karte erscheint und deshalb dauerhaft bleiben sollte, sowie einen Titel, der im Client und auf der Startseite angezeigt wird. Damit ein Projekt im Client geöffnet werden kann, muss die zugehörige Aktion aktiv sein; andernfalls bleibt das Projekt für andere Zwecke nutzbar, etwa als OWS-Dienst.

Projekte werden auf verschiedene Weise in die Applikation eingebunden: als ausdrückliche Liste, durch Angabe von Verzeichnissen, aus denen alle gefundenen Projekte geladen werden, oder über einzelne Dateipfade. Diese Wege lassen sich kombinieren.

Viele global definierte Einstellungen kann ein Projekt überschreiben oder ergänzen. So pflegen Sie eine gemeinsame Basis für alle Projekte und beschreiben je Projekt nur die Abweichungen.

%see
Siehe auch: [Konfiguration/Projekt](/admin-de/konfiguration/project), [Aktion/project](/admin-de/konfiguration/action/project).
%end

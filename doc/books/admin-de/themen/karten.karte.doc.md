# Karte :/admin-de/themen/karten/karte

Eine Karte ist eine geordnete Sammlung von Layern in einem gemeinsamen Koordinatensystem. Pro Projekt gibt es eine Hauptkarte und optional eine Übersichtskarte – eine zweite, eigenständige Karte, die im Client als Miniaturansicht einen größeren Überblick bietet.

Die Karte legt fest, welchen Bereich der Nutzer sehen darf: Ein Ausmaß begrenzt die Ausdehnung, über die hinaus nicht gescrollt werden kann; wird keines angegeben, berechnet die WebSuite es aus den Layern. Eine Anfangsposition bestimmt den Ausschnitt beim Öffnen. Über die Zoomstufen steuern Sie, wie fein und in welchen Schritten gezoomt werden kann – wahlweise als Maßstäbe oder als Auflösungen; ohne Angabe gelten Standardstufen.

Die Aktion `map` muss im Abschnitt `actions` der Konfiguration aktiviert sein. Sie rendert die Karte und verarbeitet die Interaktionen der Nutzer.

%see
Siehe auch: [Konfiguration/Karten](/admin-de/konfiguration/map), [Aktion/map](/admin-de/konfiguration/action/map).
%end

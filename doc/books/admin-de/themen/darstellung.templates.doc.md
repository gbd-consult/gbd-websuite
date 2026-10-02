# Vorlagen :/admin-de/themen/darstellung/templates

Vorlagen erzeugen dynamische Ausgaben und werden an vielen Stellen der WebSuite eingesetzt. Sie liegen in verschiedenen Formaten vor: als HTML mit der eingebauten Vorlagensprache (Jump), als reiner Text oder als Python. Neben Variablen erlauben Vorlagen einfache Programmierlogik, sodass die Ausgabe vom Kontext abhängen kann – etwa unterschiedliche Inhalte für angemeldete und anonyme Nutzer.

Wozu eine Vorlage dient, gibt ihr *Subject* an. Anhand des Subjects wählt die WebSuite die passende Vorlage für einen bestimmten Zweck; für alle Zwecke gibt es Standardvorlagen, die Sie durch eigene ersetzen können. Die wichtigsten Subjects sind:

- `application.home` – die Startseite unter `/` mit der Projektliste
- `application.error` – die Fehlerseite bei HTTP-Fehlern
- `project.home` – die Seite, die den Client für ein Projekt lädt
- `feature.title` – ein kurzer Titel eines Features für Trefferlisten und Pop-ups
- `feature.label` – die Beschriftung eines Features auf der Karte
- `feature.description` – die ausführliche Darstellung eines Features in der Infobox
- `layer.description` – die Infobox eines Layers
- `project.description` – die Infobox eines Projekts

Mit Vorlagen passen Sie Erscheinungsbild und Verhalten der Anwendung an – etwa das HTML einer Legende oder die Auswahl der Informationen, die im Pop-up eines Features erscheinen – und bilden bei Bedarf eigene Logik ab.

%see
Siehe auch: [Konfiguration/Vorlagen](/admin-de/konfiguration/template).
%end

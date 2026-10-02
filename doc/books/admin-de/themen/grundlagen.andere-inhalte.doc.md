# Andere Inhalte :/admin-de/themen/grundlagen/andere-inhalte

Nicht alles, was die GBD WebSuite ausliefert, gehört zur Karte. Startseiten, Projektseiten, Fehlerseiten, die Dateien des Clients, Logos, Stylesheets und heruntergeladene Dokumente sind gewöhnliche Web-Inhalte; dafür bringt die WebSuite einen integrierten Webserver mit.

Konfiguriert wird dies über eine oder mehrere Webseiten, die sich jeweils an einem Hostnamen festmachen; die Webseite mit dem Hostnamen `*` dient als Voreinstellung.

Jede Webseite unterscheidet zwei Arten von Inhalten. Das *web*-Verzeichnis (standardmäßig `/data/web`) enthält statische Dokumente – JavaScript, CSS, Bilder oder PDFs –, die unverändert ausgeliefert werden. Das *assets*-Verzeichnis (standardmäßig `/data/assets`) enthält Dateien, die der Server erst verarbeitet, bevor er sie ausliefert: dynamische Vorlagen ebenso wie Dateien, deren Zugriff auf bestimmte Rollen beschränkt ist.

Damit aus internen Anfragen lesbare Adressen werden, kennt jede Webseite Rewrite-Regeln, die eingehende URLs auf interne Aufrufe abbilden – und umgekehrt selbst erzeugte Adressen wieder in ihre lesbare Form bringen.

Über die einzelnen Webseiten hinaus stellt die Applikation Standardvorlagen bereit – für die Startseite, die Projektseite, Fehlerseiten und den Druck. Diese lassen sich bei Bedarf durch eigene Vorlagen ersetzen.

%see
Siehe auch: [Aktion/web](/admin-de/konfiguration/action/web), [Konfiguration/Vorlagen](/admin-de/konfiguration/template).
%end

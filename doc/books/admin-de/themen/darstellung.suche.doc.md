# Suche :/admin-de/themen/darstellung/suche

Die GBD WebSuite bietet eine einheitliche Suche über Raum- und Sachdaten. Eine Anfrage kann ein Stichwort enthalten, eine Geometrie, die die Suche räumlich einschränkt, oder beides. Auch das aus anderen GIS bekannte Identifizieren ist eine Suche – ohne Stichwort, mit einer Punktgeometrie.

Mit der Anfrage schickt der Client die zu durchsuchenden Layer mit: alle sichtbaren Layer oder, wenn ein Layer ausgewählt ist, nur diesen. Der Server befragt daraufhin die zuständigen *Finder* und führt deren Ergebnisse zu einer einheitlichen Liste von Features zusammen.

## Finder

Ein *Finder* ist ein Objekt, das eine bestimmte Quelle durchsucht. Finder können global bzw. für ein ganzes Projekt gelten oder an einem Layer hängen. Die meisten Layer bringen einen impliziten Finder mit, der ihre eigene Datenquelle durchsucht; ein Layer kann aber auch einen eigenen Finder definieren, der eine ganz andere Quelle abfragt. Bei einer Suche werden die global und im Projekt konfigurierten Finder herangezogen sowie die Finder der beteiligten Layer.

## Text- und Geometriesuche

Wie ein datenbankgestützter Finder sucht, ergibt sich aus dem zugehörigen Modell. Damit nach einem Stichwort gesucht werden kann, muss ein Textfeld des Modells für die Textsuche markiert sein; der Finder durchsucht dann dieses Feld. Für die räumliche Suche wertet der Finder das Geometriefeld des Modells aus und liefert die Features, deren Geometrie sich mit der angefragten Fläche überschneidet. Beides greift ineinander: Sind Stichwort und Geometrie angegeben, gelten beide Bedingungen zugleich.

Betrachtet eine räumliche Suche keine ausdrückliche Geometrie, bestimmt der Suchkontext, ob die ganze Karte oder nur der aktuelle Ausschnitt durchsucht wird.

%see
Siehe auch: [Konfiguration/Finder](/admin-de/konfiguration/finder), [Aktion/search](/admin-de/konfiguration/action/search).
%end

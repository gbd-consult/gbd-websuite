# ALKIS :/admin-de/themen/fachmodule/alkis

Die GBD WebSuite kann Daten aus dem Amtlichen Liegenschaftskatasterinformationssystem (ALKIS) durchsuchen und im Client eine Flurstückssuche anbieten. Voraussetzung ist eine PostgreSQL/PostGIS-Datenbank, die aus den ALKIS-Quelldaten im NAS-Format aufgebaut wurde.

Eingebunden wird ALKIS über eine Server-Aktion, die auf die Datenbank verweist und das Schema der ALKIS-Daten, ein eigenes Schema für die Indizes sowie das Koordinatensystem benennt. Bevor die Suche genutzt werden kann, müssen die Daten für die WebSuite indiziert werden; dies geschieht über die Kommandozeile und ist nach jeder ALKIS-Aktualisierung zu wiederholen. Der Index wird ausschließlich in das dafür vorgesehene Schema geschrieben, die ALKIS-Daten selbst bleiben unverändert.

Zugänglich wird die Suche im Client über das entsprechende Seitenleisten-Element. Sie lässt sich um den Zugang zu Buchungs- und Eigentümerdaten erweitern; der Zugang zu Eigentümerdaten kann auf bestimmte Rollen beschränkt und auf Wunsch protokolliert werden, wobei jeder Zugriff einen dokumentierten Grund erfordert.

%see
Siehe auch: [Aktion/alkis](/admin-de/konfiguration/action/alkis).
%end

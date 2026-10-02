# GekoS :/admin-de/themen/fachmodule/gekos

Die GBD WebSuite lässt sich mit dem Bauverwaltungssystem „GekoS Bau+" verbinden, sodass beide Systeme wechselseitig aufeinander verweisen: Aus GekoS heraus wird ein Vorgang an seiner Lage in der Karte der WebSuite angezeigt, und aus der Karte heraus lassen sich Koordinaten oder Flurstücke an GekoS zurückgeben.

Die Verbindung besteht ausschließlich aus Adressen. Beide Systeme rufen einander über URLs auf; einen Dienst, der im Hintergrund Daten abgleicht, gibt es nicht. Sie hinterlegen dafür einmal die [Adressen der WebSuite](/admin-de/konfiguration/action/gekos) in der Verfahrensadministration von GekoS und richten in der WebSuite die Aktion `gekos` ein.

## Voraussetzungen

- Die Aktion `gekos` im Projekt oder in der Applikation.
- Die Aktion `alkis` im selben Projekt. Die Rückrufe, die zu einem Flurstücks- oder Adress-Code eine Koordinate liefern, lösen diese Codes über das [ALKIS-Modul](/admin-de/themen/fachmodule/alkis) auf; fehlt die Aktion, antwortet die Schnittstelle mit `error:`.
- Das Element `Toolbar.Gekos` in der [Client-Konfiguration](/admin-de/konfiguration/client), damit die Rückgabe einer Koordinate aus der Karte funktioniert.
- Für die Übernahme von Vorgängen aus Gek-Online zusätzlich eine Datenbank mit einer Tabelle, die ausschließlich diesem Zweck dient.

## Von GekoS in die Karte

Der Sachbearbeiter hat einen Vorgang geöffnet und möchte dessen Lage sehen. GekoS öffnet die Karte mit der Koordinate oder dem Flurstückskennzeichen in der Adresse. Die WebSuite wertet diese Startparameter aus, springt auf die Stelle und setzt eine Markierung. Für die Anzeige eines Flurstücks übernimmt das ALKIS-Modul die Auflösung des Kennzeichens.

Dieser Weg braucht auf Seiten der WebSuite nichts weiter als die eingetragenen Adressen — kein Werkzeug, keine Anmeldung, sofern das Projekt öffentlich ist.

## Von der Karte zurück nach GekoS

Umgekehrt kennt GekoS manchmal die Lage noch nicht und lässt den Nutzer sie in der Karte bestimmen. Dazu öffnet es die Karte mit einer Rücksprungadresse im Parameter `gekosUrl`. Die WebSuite erkennt diesen Parameter und **startet das GekoS-Werkzeug von selbst**. Der Nutzer klickt in die Karte, ein kleiner Dialog zeigt die getroffenen Koordinaten als X und Y an und lässt sie bei Bedarf von Hand nachbessern. Mit der Bestätigung springt der Browser zurück zu der Adresse aus `gekosUrl`, ergänzt um `x` und `y`.

%info
Das Werkzeug erscheint **nur**, wenn die Karte mit `gekosUrl` in der Adresse aufgerufen wurde. Ohne diesen Parameter blendet die WebSuite die Schaltfläche `Toolbar.Gekos` aktiv aus, auch wenn sie in der Client-Konfiguration steht. Das ist beabsichtigt: Ein Rücksprung ohne Ziel wäre sinnlos. Suchen Sie die Schaltfläche also nicht beim gewöhnlichen Aufruf der Karte, sondern prüfen Sie den Weg über GekoS.
%end

## Rückrufe ohne Oberfläche

Zwei der Adressen öffnen keine Karte, sondern beantworten die Frage nach der Koordinate unmittelbar: GekoS übergibt einen Flurstücks- oder Adress-Code und erhält reinen Text im Format `x;y` zurück, im Fehlerfall `error:`. Damit kann GekoS Vorgänge räumlich verorten, ohne dass ein Mensch beteiligt ist. Auch hier ist das ALKIS-Modul die Instanz, die den Code auflöst.

## Vorgänge aus Gek-Online als Kartenebene

Bringt die Installation das Modul Gek-Online mit, kann die WebSuite dessen Vorgänge in eine räumliche Datenbanktabelle übernehmen. Aus dieser Tabelle wird anschließend ein gewöhnlicher Layer, sodass sich alle Vorgänge gemeinsam auf der Karte betrachten und durchsuchen lassen — die Umkehrung der Einzelabfrage.

Der Abgleich läuft nicht selbsttätig, sondern über den [](/admin-de/konfiguration/cli/gekos), typischerweise als geplante Aufgabe. Zwei Eigenheiten sollten Sie dabei kennen: Die Zieltabelle wird bei jedem Lauf neu angelegt, eignet sich also ausschließlich für diesen Zweck; und Vorgänge, die auf derselben Koordinate liegen, ordnet die WebSuite auf Wunsch kreisförmig an, damit sie im Client einzeln anklickbar bleiben.

## Fehlersuche

- **Die Karte öffnet, zeigt aber keine Markierung.** Prüfen Sie die Reihenfolge der Platzhalter im Flurstückscode. Eine falsche Reihenfolge führt nicht zu einer Fehlermeldung, sondern zu einem leeren Ergebnis.
- **Ein Rückruf antwortet mit `error:`.** Entweder fehlt die Aktion `alkis` im Projekt, oder der übergebene Code findet keine Entsprechung. Das Protokoll nennt beide Fälle getrennt.
- **Die Schaltfläche fehlt.** Siehe oben — ohne `gekosUrl` ist das der Normalzustand.

%see
Siehe auch: [Aktion/gekos](/admin-de/konfiguration/action/gekos), [Kommandozeilen-Befehl/gekos](/admin-de/konfiguration/cli/gekos).
%end

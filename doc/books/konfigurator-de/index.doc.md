# Konfigurator-Handbuch :/konfigurator-de

Der GBD Konfigurator ist die grafische Oberfläche zum Anzeigen und Bearbeiten der Konfiguration einer GBD WebSuite Installation. Die Oberfläche wird aus dem aktuell geladenen GWS Schema erzeugt. Dadurch erscheinen nur die Bereiche, Objekttypen und Eigenschaften, die von der verbundenen Installation unterstützt werden.

Dieses Handbuch richtet sich an Administratoren und Projektverantwortliche. Es erklärt den Arbeitsbereich, die Navigation, die verfügbaren Editoren sowie den sicheren Umgang mit lokalen Änderungen, Projekten, JSON und Berechtigungen.

%warn
**Wichtig:** Eine Bearbeitung verändert zunächst nur den lokalen Entwurf im Browser. Erst nach dem abschließenden Bestätigen mit **Anwenden** wird die Konfiguration an den Server gesendet.
%end

## Sicher starten

Für eine kontrollierte Änderung empfiehlt sich immer dieselbe Reihenfolge:

1.  Den gewünschten Bereich oder das gewünschte Projekt im Navigationsbaum öffnen.
2.  Den aktuellen Wert und die Beschreibung im Leitfaden prüfen.
3.  Den Wert im passenden Editor bearbeiten.
4.  Die lokalen Änderungen im Änderungsbereich kontrollieren.
5.  Berechtigungen, Objekt-UIDs und mögliche Löschungen gesondert prüfen.
6.  Die Konfiguration erst nach dieser Kontrolle auf den Server anwenden.
7.  Den betroffenen Bereich anschließend erneut öffnen und das Ergebnis kontrollieren.

Nicht gespeicherte Änderungen bleiben im Browser als lokaler Entwurf sichtbar. Sie können einzeln oder vollständig verworfen werden, ohne den aktuellen Serverstand zu verändern.

## Arbeitsbereich

Der Arbeitsbereich besteht aus mehreren dauerhaft miteinander verbundenen Bereichen:

-   **Bereichsleiste ①:** Wechsel zwischen den fachlichen Hauptbereichen der Konfiguration.
-   **Konfigurationsbaum ②:** Hierarchische Navigation durch App, Projekte und weitere Konfigurationsobjekte.
-   **Kopfzeile ③:** Zugriff auf Berechtigungen, Aktualisieren, JSON Ansicht und Farbschema.
-   **Editor ④:** Zeigt die Eigenschaften und passenden Eingabeelemente des ausgewählten Objekts.
-   **Leitfaden ⑤:** Erklärt Bedeutung, Pflichtstatus und Typ des aktuellen Bereichs oder Feldes.
-   **Pfadnavigation ⑥:** Zeigt die Position des aktuell geöffneten Objekts.
-   **Änderungsbereich [⎋](/konfigurator-de/lokale-aenderungen-pruefen):** Zeigt alle noch nicht veröffentlichten Änderungen.

![Arbeitsbereich des GBD Konfigurators](/konfigurator-images/01-arbeitsbereich.png)

*Gesamtansicht mit Bereichsleiste, Konfigurationsbaum, Editor und Leitfaden.*

**Einen Bereich auswählen**

Die Symbole am linken Rand öffnen die verfügbaren Hauptbereiche. Welche Einträge erscheinen, hängt von der geladenen Konfiguration und den Berechtigungen des angemeldeten Benutzers ab.

Nach der Auswahl eines Bereichs wird dessen Wurzel im Konfigurationsbaum geöffnet. Der Editor zeigt gleichzeitig das zugehörige Objekt an.

**Ein Objekt öffnen**

Ein Pfeil vor einem Eintrag kennzeichnet einen Knoten mit untergeordneten Elementen. Durch Auswahl der Zeile wird das Objekt im Editor geöffnet. Durch Auswahl des Pfeils wird der Knoten ein- oder ausgeklappt.

Wenn ein Objekttyp nicht durch einen spezialisierten Editor unterstützt wird, zeigt der Konfigurator einen entsprechenden Hinweis. Die Anwendung wird dadurch nicht beendet.

## Navigation und Suche

**Konfigurationsbaum**

Der Konfigurationsbaum bildet die hierarchische Struktur der geladenen GWS Konfiguration ab. Einträge können geöffnet, geschlossen und direkt ausgewählt werden.

Die Breite des Navigationsbereichs kann angepasst werden. Eine größere Breite ist hilfreich, wenn lange Objektbezeichnungen oder tief verschachtelte Pfade angezeigt werden.

**Im Baum suchen**

Das Suchfeld **①** oberhalb des Baums durchsucht die sichtbaren Einträge. Die zugehörigen Suchoptionen ermöglichen eine genauere Eingrenzung:

-   **Aa:** Groß- und Kleinschreibung berücksichtigen.
-   **W:** Nur vollständige Wörter berücksichtigen.
-   **Enter:** Zum nächsten Treffer wechseln.
-   **Umschalt + Enter:** Zum vorherigen Treffer wechseln.
-   **Escape:** Suche schließen oder Suchbegriff zurücksetzen.

Der gefilterte Konfigurationsbaum **②** zeigt die passenden Treffer und ihre Position in der Hierarchie.

![Suche im Konfigurationsbaum](/konfigurator-images/02-baumsuche.png)

*Die Suche filtert den Baum und zeigt die Anzahl der gefundenen Einträge.*

**Pfadnavigation**

Die Pfadnavigation oberhalb des Editors zeigt die Position des aktuellen Objekts. Ein übergeordneter Eintrag im Pfad kann direkt ausgewählt werden, um eine höhere Ebene zu öffnen.

Die Pfeile für Zurück und Vorwärts verwenden den internen Navigationsverlauf des Konfigurators. Sie ändern nicht den Verlauf des gesamten Browserfensters.

## Leitfaden

Der Leitfaden liefert kontextbezogene Informationen zum aktuellen Bereich oder Feld. Je nach Schema enthält er:

-   eine fachliche Beschreibung,
-   den Pflichtstatus,
-   Hinweise zu optionalen und bereits gesetzten Werten,
-   den Namen oder die Referenz des zugrunde liegenden Typs,
-   einen Verweis auf die vollständige Typdokumentation.

**Leitfaden positionieren**

Der Leitfaden kann an unterschiedlichen Stellen angezeigt werden:

-   links unterhalb des Konfigurationsbaums,
-   rechts als Seitenpanel **①**,
-   eingeklappt, wenn mehr Platz für den Editor benötigt wird.

Je nach Ansicht lassen sich Breite oder Höhe über den Trenner anpassen. Über das mit **②** markierte Andocksymbol kann die Position des Leitfadens gewechselt werden. Position, Größe und Öffnungszustand werden lokal im Browser gespeichert.

![Rechts angedockter Leitfaden](/konfigurator-images/03-leitfaden-rechts.png)

*Der Leitfaden kann rechts als eigenes Panel angezeigt werden.*

## Werte und Felder bearbeiten

Die sichtbaren Editoren werden aus dem geladenen GWS Schema erzeugt. Pflichtfelder sind mit einem Stern gekennzeichnet. Änderungen werden unmittelbar als lokale, noch nicht veröffentlichte Änderungen erfasst.

### Text und Boolesche Werte
**Text und Zahlen**

Textwerte werden direkt in einem Eingabefeld bearbeitet. Zahlenfelder unterscheiden je nach Schema zwischen Ganzzahlen und Dezimalzahlen. Ungültige Werte werden markiert und können nicht fehlerfrei angewendet werden.

**Boolesche Werte**

Boolesche Eigenschaften werden als Kontrollkästchen oder Schalter dargestellt. Der sichtbare Zustand entspricht dem aktuell im lokalen Entwurf gespeicherten Wert.

![Textfeld und boolesche Werte](/konfigurator-images/12-text-und-boolesch.png){border=1, width=560px}

*Beispiel für ein Textfeld und mehrere aktivierte boolesche Optionen in einer realen Serverkonfiguration.*

### Variantentypen

Bei einem Variantentyp bestimmt die Typauswahl, welche weiteren Felder sichtbar sind. Nach einem Typwechsel können neue Pflichtfelder erscheinen oder bisher sichtbare Felder entfallen.

%info
**Achtung:** Prüfen Sie nach einem Typwechsel alle neu eingeblendeten Pflichtfelder und die Änderungsliste, bevor Sie den Entwurf anwenden.
%end

![Typauswahl eines Variantentyps](/konfigurator-images/14-variantentyp.png){border=1, width=520px}

*Die Typauswahl bestimmt, welche zugehörigen Felder eingeblendet und gegebenenfalls verpflichtend werden.*

### Zeitdauer

Zeitspannen werden in getrennten Feldern für Tage, Stunden, Minuten und Sekunden bearbeitet. Der Konfigurator setzt die Eingaben in den vom Schema erwarteten Wert um.

![Eingabefelder für eine Zeitdauer](/konfigurator-images/13-zeitdauer-standardwert.png){border=1, width=520px}

*Eine Zeitdauer wird auf mehrere Eingabefelder verteilt; das Rücksetzsymbol stellt den definierten Standardwert wieder her.*

### Koordinatenreferenzsystem

Ein Koordinatenreferenzsystem kann über einen EPSG Code eingegeben oder aus häufig verwendeten Systemen ausgewählt werden. Nach einer Änderung sollten Center, Extent und Kartenvorschau gemeinsam geprüft werden.

![Auswahl eines Koordinatenreferenzsystems](/konfigurator-images/15-koordinatenreferenzsystem.png){border=1, width=520px}

*Der EPSG-Code kann direkt eingegeben oder aus häufig verwendeten Koordinatenreferenzsystemen ausgewählt werden.*

### Standardwert wiederherstellen

Wenn das Schema einen Standardwert definiert, kann ein geänderter Wert über das Rücksetzsymbol wieder auf diesen Standard gesetzt werden. Das Zurücksetzen ist ebenfalls eine lokale Änderung und muss anschließend geprüft werden.

### Auswahlfelder

Ein Auswahlfeld erlaubt ausschließlich die im Schema definierten Werte. Dadurch werden ungültige freie Eingaben vermieden.

### Kartenmittelpunkt

Der Kartenmittelpunkt besteht aus X- und Y-Koordinate. Die Werte können direkt eingegeben werden. Wenn die Kartenvorschau verfügbar ist, kann der Mittelpunkt auch durch einen Klick auf die Karte oder durch Verschieben des Markers gesetzt werden.

![X- und Y-Koordinate des Kartenmittelpunkts](/konfigurator-images/16-kartenmittelpunkt.png){border=1, width=340px}

*Direkte Eingabe der X- und Y-Koordinate des Kartenmittelpunkts.*

### Räumliche Ausdehnung

Der Extent definiert einen rechteckigen Darstellungsbereich. Er besteht aus minimalen und maximalen X- und Y-Koordinaten.

![Koordinaten einer räumlichen Ausdehnung](/konfigurator-images/17-raeumliche-ausdehnung.png){border=1, width=340px}

*Der Extent wird durch minimale und maximale X- und Y-Werte begrenzt.*

Eine räumliche Ausdehnung kann auf zwei Arten bearbeitet werden:

1.  Werte direkt in die Koordinatenfelder eingeben.
2.  **Extent bearbeiten** auswählen.
3.  Ein Rechteck auf der Karte aufziehen.
4.  Die erzeugten Minimal- und Maximalwerte bei Bedarf feinjustieren.

Ein leerer optionaler Extent bedeutet, dass kein eigener Darstellungsbereich gesetzt ist. In diesem Fall wird der übergeordnete oder systemseitige Bereich verwendet.

### Kontrolle räumlicher Änderungen

Nach einer Änderung sollten mindestens folgende Punkte geprüft werden:

-   passt das CRS zu den Koordinaten,
-   liegt der Marker an der erwarteten Position,
-   enthält das Rechteck den gewünschten Bereich,
-   sind minimale Werte kleiner als die zugehörigen maximalen Werte,
-   bleibt die Kartenvorschau sichtbar und plausibel.

### Einfache Listen

Listen aus skalaren Werten werden zeilenweise bearbeitet. Abhängig vom Feld können Texte, Zahlen oder andere einfache Werte hinzugefügt werden.

![Einfache Liste mit skalaren Werten](/konfigurator-images/19-einfache-liste.png){border=1, width=520px}

*Einträge können ergänzt, über den Ziehgriff neu sortiert oder über das Kreuz entfernt werden.*

### Komplexe Listen

Komplexe Listen enthalten vollständige Objekte. Ein Element kann geöffnet werden, um seine Unterfelder zu bearbeiten.

![Komplexe Liste mit Kartenlayern](/konfigurator-images/20-komplexe-liste.png){border=1, width=560px}

*Beispiel einer komplexen Liste: Jeder Layer ist ein eigenes Objekt und kann geöffnet, verschoben oder entfernt werden.*

Typische Aktionen sind:

-   Element hinzufügen,
-   Element öffnen,
-   Element über den Ziehgriff neu anordnen,
-   Element über das Kreuz entfernen.

%info
**Achtung:** Die Reihenfolge kann fachlich relevant sein. Das gilt insbesondere für Regeln, Prioritäten und Verarbeitungsketten.
%end

### Wörterbücher

Ein Wörterbuch ordnet eindeutigen Schlüsseln jeweils einen Wert zu. Schlüssel und Werte werden tabellarisch gepflegt. Doppelte Schlüssel sind nicht zulässig.

![Wörterbuch mit Schlüssel-Wert-Paaren](/konfigurator-images/21-woerterbuch.png){border=1, width=520px}

*Schlüssel und zugehörige Werte werden zeilenweise erfasst und können neu angeordnet oder entfernt werden.*

### Freie strukturierte Werte

Einige Felder erlauben eine freiere strukturierte Eingabe. Der zulässige Inhalt wird weiterhin durch das Schema und die Validierung des Konfigurators begrenzt.

![Freier strukturierter JSON-Wert](/konfigurator-images/18-freier-json-wert.png){border=1, width=520px}

*Freie strukturierte Werte werden direkt als JSON bearbeitet und anschließend validiert.*

## Projekte verwalten

Der Bereich **projects** enthält die in der GBD WebSuite verfügbaren Projekte. **Die Projektliste ①** zeigt alle vorhandenen Projekte in ihrer aktuellen Reihenfolge. Über **die Ziehgriffe ②** lassen sich Projekte neu anordnen. **Die Symbole ③** dienen dazu, ein Projekt als Vorlage zu speichern oder es zu entfernen. Jedes Projekt besitzt eine eindeutige UID und eine eigene verschachtelte Konfiguration. 

Die Schaltfläche **Projekt erstellen ④** öffnet die Auswahl, auf die weiter unten näher eingegangen wird.

![Projektübersicht](/konfigurator-images/08-projekte.png)

*Die Projektübersicht zeigt alle vorhandenen Projekte sowie ihre Reihenfolge.*

### Projekt erstellen

Über **Projekt erstellen ①** stehen abhängig vom aktuellen Zustand folgende Möglichkeiten zur Verfügung **②**:

-   **Leeres Projekt erstellen:** Legt einen neuen Ausgangspunkt an.
-   **Vorhandenes Projekt kopieren:** Erstellt ein vollständiges Duplikat als Grundlage.
-   **Projekt einfügen:** Erstellt ein Projekt aus einem vorbereiteten JSON Objekt.
-   **Aus Vorlage erstellen:** Verwendet ein zuvor als Vorlage markiertes Projekt.

![Menü zum Erstellen eines Projekts](/konfigurator-images/09-projekt-erstellen.png)

*Ein Projekt kann leer, als Kopie, aus JSON oder aus einer Vorlage erstellt werden.*

### Projektvorlage verwenden

Ein Projekt kann über das Lesezeichensymbol als Vorlage markiert werden. Die Auswahl wird lokal im Browser gespeichert. Neue Kopien erhalten eine eigene UID und bleiben zunächst unveröffentlicht.

### Projekt aus JSON einfügen

Beim Einfügen eines Projekts wird ein vollständiges JSON Objekt in den Dialog kopiert.

Vorgehen:

1.  Vollständiges JSON Objekt in das Eingabefeld einfügen.
2.  Syntaxfehler korrigieren.
3.  Doppelte Objekt-UIDs korrigieren.
4.  **Projekt hinzufügen** auswählen.
5.  Das neue Projekt im Konfigurationsbaum öffnen.
6.  Die erzeugten lokalen Änderungen kontrollieren.

Fehlt die UID des neuen Projektobjekts, kann der Konfigurator eine neue UID vergeben. Untergeordnete doppelte UIDs müssen dennoch korrigiert werden.

## Lokale Änderungen prüfen

Jede Bearbeitung wird zunächst als lokale Änderung gespeichert. **Der Änderungsbereich ①** zeigt alle Abweichungen vom geladenen Serverstand. **Die** **Anzeige** **③** in der Kopfzeile zeigt die Anzahl der lokalen Änderungen und öffnet den Änderungsbereich.

![Nicht veröffentlichte Änderungen](/konfigurator-images/10-lokale-aenderungen.png)

*Der Änderungsbereich stellt Ausgangswert und neuen lokalen Wert gegenüber.*

**Der Änderungseintrag ②** stellt Ausgangswert und neuen Wert gegenüber und bietet **Öffnen** sowie **Verwerfen** an. Zu jedem Eintrag können abhängig von der Änderungsart folgende Informationen erscheinen:

-   Pfad des geänderten Objekts oder Feldes,
-   Art der Änderung,
-   ursprünglicher Wert,
-   neuer Wert,
-   neu erstelltes oder gelöschtes Objekt.

**Änderung öffnen**

Mit **Öffnen** springt der Konfigurator direkt zur betroffenen Stelle. Dadurch kann die Änderung im fachlichen Zusammenhang kontrolliert werden.

**Einzelne Änderung verwerfen**

Mit **Verwerfen** wird nur der ausgewählte Eintrag auf den geladenen Serverstand zurückgesetzt.

**Alle Änderungen verwerfen**

Die Funktion zum vollständigen Verwerfen löscht den gesamten lokalen Entwurf. Der Serverstand bleibt unverändert.

## Änderungen anwenden oder verwerfen

Die Schaltfläche **Aktualisieren** öffnet zunächst eine Sicherheitsabfrage. Erst die Bestätigung mit **Anwenden** sendet den lokalen Entwurf an den Server.

Vor dem Anwenden sollten folgende Punkte geprüft werden:

-   Anzahl der Änderungen,
-   neu erstellte Objekte,
-   gelöschte Objekte,
-   geänderte UIDs,
-   geänderte Berechtigungen,
-   neu entstandene Pflichtfeldfehler,
-   räumliche Werte und Kartenvorschau.

**Fehler beim Anwenden**

Schlägt das Anwenden fehl, bleibt der lokale Entwurf erhalten. Die Fehlermeldung sollte vollständig gelesen werden, bevor weitere Änderungen vorgenommen werden.

Nach einem Fehler:

1.  Fehlermeldung lesen.
2.  Betroffenen Pfad oder Wert ermitteln.
3.  Eingabe, UID oder Berechtigung korrigieren.
4.  Änderungsliste erneut prüfen.
5.  Anwenden erneut bestätigen.

## Vollständige JSON Konfiguration

**Die vollständige JSON Ansicht ①** zeigt den normalisierten Stand der geladenen Konfiguration.

![Vollständige JSON Konfiguration](/konfigurator-images/06-json-konfiguration.png)

*Im Nur-Lesen-Modus kann der vollständige Stand kontrolliert und kopiert werden.*

**Nur lesen**

**Der Status** **②** *"Nur lesen"* kennzeichnet den schreibgeschützten Anzeigemodus. In diesem Modus kann die Konfiguration durchsucht und kopiert werden. Er eignet sich zur Kontrolle, Fehlersuche und Weitergabe eines vollständigen Stands.

**Bearbeitungsmodus**

Mit **der Schaltfläche** **③** \*"\**Bearbeitung aktivieren"* wird der Bearbeitungsmodus geöffnet. In diesem Modus kann die vollständige Konfiguration als JSON geändert werden.

**Übernehmen** speichert den bearbeiteten JSON Inhalt vorläufig im lokalen Entwurf. Erst **Aktualisieren** und **Anwenden** senden ihn an den Server.

Der Konfigurator weist unter anderem folgende Fehler zurück:

-   ungültige JSON Syntax,
-   doppelte Objekt-UIDs,
-   Werte mit einem falschen Datentyp,
-   nicht auflösbare Objektstrukturen,
-   fehlende Pflichtfelder.

%info
**Achtung:** Der JSON Bearbeitungsmodus kann viele Bereiche gleichzeitig verändern. Verwenden Sie ihn nur, wenn die Auswirkungen des vollständigen Objekts bekannt sind.
%end

## Berechtigungen

**Die Berechtigungshierarchie ①** ordnet die gefundenen Konfigurationsobjekte hierarchisch an.  
Die mit **②** markierten Badges zeigen die vom Server berechneten effektiven Rollen für Lesen, Schreiben, Erstellen und Löschen.

![Berechtigungsübersicht ohne lokale Änderungen](/konfigurator-images/04-berechtigungen.png)

*Wenn keine lokalen Änderungen vorliegen, werden die effektiven Rollen als Badges angezeigt.*

%warn
**Wichtig bei nicht veröffentlichten Änderungen:** Sobald ein lokaler Entwurf vorhanden ist, kann der Konfigurator die effektiven Berechtigungen nicht mehr zuverlässig anzeigen. Die Berechtigungsdaten stammen vom Server und gehören zum zuletzt geladenen Serverstand, während der lokale Entwurf bereits davon abweicht. Deshalb erscheint **der Hinweis ①** *„Änderungen übernehmen oder verwerfen.“* und die serverberechneten Rollen werden ausgeblendet.
%end

![Berechtigungsübersicht bei nicht veröffentlichten Änderungen](/konfigurator-images/11-berechtigungen-gesperrt.png)

*Bei lokalen Änderungen sind die effektiven Rollen vorübergehend nicht verfügbar.*

Die effektiven Berechtigungen werden wieder sichtbar, wenn eine der folgenden Bedingungen erfüllt ist:

1.  Die lokalen Änderungen werden vollständig verworfen. Danach entspricht die sichtbare Konfiguration wieder dem geladenen Serverstand.
2.  Die Änderungen werden über **die markierte Schaltfläche ③** *"Aktualisieren"* und anschließend **Anwenden** gespeichert. Danach wartet der Konfigurator auf die neu berechneten Berechtigungsdaten des Servers.

Während dieser Zeit bleiben die rohen ACL Regeln **②** wie `allow all` oder `allow admin-1` sichtbar und können bearbeitet werden. Sie dürfen jedoch nicht mit den effektiven, vom Server berechneten Rollen verwechselt werden. Besonders geerbte Rechte lassen sich erst nach dem Verwerfen oder erfolgreichen Anwenden wieder vollständig beurteilen.

### ACL Regeln bearbeiten

ACL Regeln legen fest, welche Rollen eine Aktion ausführen dürfen oder nicht ausführen dürfen. Die Regeln werden in einer definierten Reihenfolge ausgewertet.

![Editor für ACL-Regeln](/konfigurator-images/22-acl-regeln.png){border=1, width=520px}

*Jede ACL-Regel enthält eine Entscheidung, eine Rolle sowie Steuerelemente zum Sortieren und Entfernen.*

Für eine Regel werden typischerweise folgende Schritte ausgeführt:

1.  **Erlauben** oder **Verweigern** wählen.
2.  Rollenname eintragen oder auswählen.
3.  Regel an die fachlich richtige Position verschieben.
4.  Weitere Regeln hinzufügen oder vorhandene Regeln entfernen.

%warn
**Wichtig:** Die erste passende Regel entscheidet. Eine falsche Reihenfolge kann deshalb ein anderes Ergebnis erzeugen als erwartet.
%end

Änderungen an der App-Wurzel werden zusätzlich geprüft, damit notwendige Zugriffe nicht vollständig entzogen werden. Eine erkannte vollständige Aussperrung blockiert das Anwenden.

### Berechtigungsübersicht

Die Berechtigungsübersicht stellt die effektiven Berechtigungen für alle gefundenen Konfigurationsobjekte hierarchisch gegenüber.

Die Übersicht hilft bei folgenden Fragen:

-   Welche Rollen dürfen ein Objekt lesen?
-   Welche Rollen dürfen vorhandene Werte bearbeiten?
-   Welche Rollen dürfen untergeordnete Objekte erstellen?
-   Welche Rollen dürfen ein Objekt löschen?
-   Welche Rechte wurden geerbt?
-   An welcher Stelle geht ein zuvor vorhandener Zugriff verloren?

Nach Änderungen an ACL Regeln sollte die Berechtigungsübersicht erneut geöffnet werden.

## Darstellung und responsive Nutzung

**Heller und dunkler Modus**

Über das mit **①** markierte Symbol für Sonne oder Mond kann zwischen hellem und dunklem Modus gewechselt werden.

![GBD Konfigurator im dunklen Modus](/konfigurator-images/07-dunkler-modus.png)

*Der dunkle Modus passt Oberfläche, Statusfarben und Eingabeelemente gemeinsam an.*

Die Auswahl wird lokal im Browser gespeichert. Sie verändert keine Konfigurationswerte und wird nicht an den Server gesendet.

**Schmale Bildschirme**

Auf schmalen Bildschirmen wird der Navigationsbaum als Overlay dargestellt. Nach der Auswahl eines Eintrags kann sich das Panel automatisch schließen, damit mehr Platz für den Editor bleibt.

Der Leitfaden passt sich an die verfügbare Breite an. Für umfangreiche Tabellen, JSON Inhalte oder Berechtigungsübersichten wird dennoch ein größerer Bildschirm empfohlen.

### Tastaturbedienung

| Taste | Funktion |
| --- | --- |
| `Enter` | Zum nächsten Suchtreffer wechseln |
| `Umschalt + Enter` | Zum vorherigen Suchtreffer wechseln |
| `Escape` | Suche oder geöffneten Dialog schließen |
| `Pfeil nach oben` / `Pfeil nach unten` | Benachbarte Zeile auswählen |
| `Leertaste` | Ausgewählten Baumknoten ein- oder ausklappen |

## Fehler einordnen

- [Konfiguration kann nicht geladen werden](./konfiguration-kann-nicht-geladen-werden)
- [Ungültiges JSON](./ungueltiges-json)
- [Doppelte UID](./doppelte-uid)
- [Pfad kann nicht aufgelöst werden](./pfad-kann-nicht-aufgeloest-werden)
- [Kartenvorschau fehlt](./kartenvorschau-fehlt)
- [Berechtigungsanzeige ist veraltet](./berechtigungsanzeige-ist-veraltet)
- [Anwenden wird wegen Aussperrung blockiert](./anwenden-wird-wegen-aussperrung-blockiert)
### Konfiguration kann nicht geladen werden

Mögliche Ursachen:

-   Server ist nicht erreichbar,
-   Sitzung ist abgelaufen,
-   notwendige Leseberechtigung fehlt,
-   Konfiguration oder Schema ist ungültig.

Verbindung, Anmeldung und Berechtigungen prüfen und die Seite anschließend neu laden.

### Ungültiges JSON

Syntaxfehler im JSON Editor korrigieren. Besonders auf fehlende Kommas, Anführungszeichen und schließende Klammern achten.

### Doppelte UID

Jedes Konfigurationsobjekt benötigt eine eindeutige UID. Die gemeldeten Objekte öffnen und einer Kopie oder einem neu eingefügten Objekt eine neue UID zuweisen.

### Pfad kann nicht aufgelöst werden

Einen gültigen übergeordneten Eintrag im Navigationsbaum öffnen. Prüfen, ob das Objekt zwischenzeitlich gelöscht, verschoben oder durch einen Typwechsel ersetzt wurde.

### Kartenvorschau fehlt

CRS, Koordinaten, Kartenquelle und notwendige Berechtigungen prüfen. Ungültige Koordinaten oder eine nicht erreichbare Quelle können die Vorschau verhindern.

### Berechtigungsanzeige ist veraltet

Lokale Änderungen entweder anwenden oder verwerfen und die Berechtigungsübersicht anschließend erneut öffnen.

### Anwenden wird wegen Aussperrung blockiert

Die Regeln an der App-Wurzel prüfen. Mindestens die für Administration und Wiederherstellung erforderlichen Rollen müssen weiterhin Zugriff besitzen.

## Empfohlener Arbeitsablauf

1.  Fachlichen Bereich oder Projekt öffnen.
2.  Beschreibung und aktuellen Wert im Leitfaden lesen.
3.  Änderung im passenden Editor durchführen.
4.  Betroffene Unterfelder und Pflichtfelder kontrollieren.
5.  Lokalen Entwurf im Änderungsbereich prüfen.
6.  Berechtigungen, UIDs und Löschungen gesondert prüfen.
7.  Änderungen über **Aktualisieren** und **Anwenden** bestätigen.
8.  Serverantwort abwarten.
9.  Betroffenen Bereich erneut öffnen.
10.  Ergebnis und Berechtigungsübersicht kontrollieren.

Dieser Ablauf reduziert das Risiko unbeabsichtigter Änderungen und macht auch umfangreiche Bearbeitungen nachvollziehbar.

## Feldtypen und Symbole

Welche Feldtypen tatsächlich angezeigt werden, bestimmt das aktuell geladene GWS Schema.

| Gruppe | Beispiele |
| --- | --- |
| Primitive Werte | Text, Ganzzahl, Dezimalzahl, Boolescher Wert, Any |
| Auswahl | Enum, Variant |
| Zeit | Duration |
| Raumbezug | CRS, Point, Extent |
| Werte mit Einheit | UOM Value, Size, Extent |
| Zugriff | ACL String und ACL Regeln |
| Sammlungen | List, Dict, Set |
| Spezialisierte Bereiche | Projects, Map, Permissions |

Häufig verwendete Statusanzeigen:

| Anzeige | Bedeutung |
| --- | --- |
| Stern | Pflichtfeld laut Schema |
| Hervorgehobene Zeile | Aktive Auswahl |
| Häkchen | Vorgang erfolgreich abgeschlossen |
| Ausrufezeichen oder Fehlermeldung | Server-, Verbindungs- oder Validierungsfehler |
| Geschweifte Klammern | Vollständige JSON Ansicht |
| Mond oder Sonne | Wechsel des Farbschemas |
| Schloss | Berechtigungen oder Berechtigungsübersicht |
| Ziehgriff | Element verschieben oder Bereich skalieren |

Der Status **Unveröffentlicht** bedeutet, dass ein Wert lokal geändert wurde, aber noch nicht über **Aktualisieren** und **Anwenden** an den Server gesendet wurde.

## Begriffe

- [Schema](./schema)
- [Konfiguration](./konfiguration)
- [UID](./uid)
- [ACL](./acl)
- [CRS](./crs)
- [Extent](./extent)
- [Lokaler Entwurf](./lokaler-entwurf)
- [Anwenden](./anwenden)

### Schema

Beschreibung der verfügbaren Konfigurationstypen, Eigenschaften, Pflichtfelder, Standardwerte und zulässigen Werte. Der Konfigurator erzeugt die sichtbaren Editoren aus diesen Daten.

### Konfiguration

Verschachteltes JSON Dokument, das den Zustand einer GBD WebSuite Installation beschreibt.

### UID

Eindeutige Kennung eines Konfigurationsobjekts. Doppelte UIDs sind innerhalb einer gültigen Konfiguration nicht zulässig.

### ACL

Geordnete Regeln, die festlegen, welche Rollen eine Aktion erlauben oder verweigern.

### CRS

Koordinatenreferenzsystem, das die Bedeutung räumlicher Koordinaten festlegt. Häufig wird es über einen EPSG Code angegeben.

### Extent

Rechteckige räumliche Ausdehnung aus minimaler und maximaler X- und Y-Koordinate.

### Lokaler Entwurf

Gesamtheit aller Änderungen, die im Browser vorgenommen, aber noch nicht an den Server gesendet wurden.

### Anwenden

Abschließender, bestätigungspflichtiger Schritt, der den geprüften lokalen Entwurf an den Server sendet.
# Konfigurator-Handbuch :/konfigurator-de

Mit dem GBD Konfigurator richten Sie eine GBD WebSuite Installation über eine grafische Oberfläche ein: Projekte, Karten und Layer, Benutzeranmeldung, Datenbankverbindungen, Dienste und Zugriffsrechte. Die Konfigurationsdateien müssen Sie dafür nicht von Hand bearbeiten.

Der Konfigurator passt sich an die verbundene Installation an. Er zeigt nur die Bereiche und Einstellungen, die Ihre Version der GBD WebSuite tatsächlich kennt. Grundlage dafür ist das sogenannte Schema, eine Beschreibung aller zulässigen Einstellungen, die die GBD WebSuite selbst mitliefert.

Dieses Handbuch richtet sich an Fachanwenderinnen und Fachanwender, die sich in die Administration der GBD WebSuite einarbeiten. Kenntnisse über Geodaten, Koordinatensysteme und WebGIS werden vorausgesetzt, Programmierkenntnisse nicht. Das Handbuch erklärt den Aufbau der Oberfläche, die Eingabefelder für die verschiedenen Arten von Einstellungen und den Weg, auf dem eine Änderung geprüft und auf den Server übertragen wird.
%warn
**Wichtiger Hinweis: Demonstrator-Version**

Beim GBD WebSuite Konfigurator handelt es sich aktuell um einen reinen Demonstrator, der noch nicht für den produktiven Einsatz freigegeben ist. Wenn Sie Interesse daran haben, den Konfigurator einzusetzen, wenden Sie sich bitte an die [Geoinformatikbüro Dassau GmbH](https://www.gbd-consult.de/).
%end
%warn
**Wichtig:** Alles, was Sie bearbeiten, wird zunächst nur in Ihrem Browser als Entwurf festgehalten. Die laufende GBD WebSuite bleibt davon unberührt. Erst wenn Sie **Aktualisieren** wählen und die Rückfrage mit **Anwenden** bestätigen, wird die Konfiguration an den Server übertragen.
%end

## Sicher starten

Bewährt hat sich, jede Änderung in derselben Reihenfolge vorzunehmen:

1.  Den gewünschten Bereich oder das gewünschte Projekt im Navigationsbaum links öffnen.
2.  Den aktuellen Wert ansehen und im Leitfaden nachlesen, wofür die Einstellung zuständig ist.
3.  Den Wert ändern.
4.  Im Änderungsbereich kontrollieren, welche Änderungen sich angesammelt haben.
5.  Besonders genau hinsehen bei geänderten Objektkennungen (UIDs), gelöschten Objekten und geänderten Zugriffsregeln.
6.  Erst danach die Konfiguration auf den Server übertragen.
7.  Den geänderten Bereich noch einmal öffnen und das Ergebnis prüfen. Nach Änderungen an Zugriffsregeln zusätzlich in der Berechtigungsübersicht kontrollieren, wer jetzt was darf.

Solange Sie nicht übertragen haben, bleibt Ihr Entwurf im Browser erhalten. Sie können einzelne Änderungen oder den ganzen Entwurf jederzeit verwerfen. Auf dem Server ändert sich dadurch nichts.

## Arbeitsbereich

Die Oberfläche gliedert sich in folgende Bereiche:

-   **Bereichsleiste ①:** Symbole am linken Rand, mit denen Sie zwischen den Hauptthemen wechseln, etwa Projekte, Anmeldung, Datenbanken oder Server.
-   **Konfigurationsbaum ②:** Alle Einstellungen der Installation als aufklappbare Baumstruktur, ähnlich dem Layerbaum in einem GIS.
-   **Kopfzeile ③:** Schaltflächen für die Berechtigungsübersicht, das Übertragen von Änderungen (**Aktualisieren**), die JSON-Ansicht und den Wechsel zwischen hellem und dunklem Farbschema.
-   **Editor ④:** Die Einstellungen des ausgewählten Eintrags mit den passenden Eingabefeldern.
-   **Leitfaden ⑤:** Erklärt, wofür das gerade gewählte Feld dient, ob es ausgefüllt werden muss und welche Art von Wert erwartet wird.
-   **Pfadnavigation ⑥:** Zeigt, wo in der Baumstruktur Sie sich gerade befinden.
-   **Änderungsbereich [⎋](/konfigurator-de/lokale-aenderungen-pruefen):** Listet alle Änderungen, die noch nicht auf den Server übertragen wurden.

![Arbeitsbereich des GBD Konfigurators](/konfigurator-images/01-arbeitsbereich.png){border=4}

*Gesamtansicht mit Bereichsleiste, Konfigurationsbaum, Editor und Leitfaden.*

## Einen Bereich auswählen ::

Über die Symbole am linken Rand öffnen Sie die Hauptthemen der Konfiguration. Welche Symbole erscheinen, hängt davon ab, was in Ihrer Installation eingerichtet ist und welche Version der GBD WebSuite läuft.

Wenn Sie ein Symbol wählen, springt der Konfigurationsbaum an den Anfang dieses Themas, und der Editor zeigt die zugehörigen Einstellungen.

## Ein Objekt öffnen ::

Einträge mit einem Pfeil davor enthalten weitere Einträge. Ein Klick auf den Eintrag selbst öffnet ihn im Editor, ein Klick auf den Pfeil klappt ihn auf oder zu.

Für einige Arten von Einstellungen gibt es noch keine eigene Eingabemaske. In diesem Fall zeigt der Konfigurator einen Hinweis an; Sie können normal weiterarbeiten.

## Navigation und Suche

## Konfigurationsbaum ::

Der Konfigurationsbaum zeigt den Aufbau Ihrer GBD WebSuite Konfiguration: von den globalen Einstellungen der Installation über die einzelnen Projekte bis hinunter zu Karten, Layern und deren Details. Sie können Einträge auf- und zuklappen und direkt anwählen.

Die Breite des Baums lässt sich mit der Maus verändern. Das ist hilfreich bei langen Namen oder tief verschachtelten Einträgen.

## Im Baum suchen ::

Mit dem Suchfeld **①** über dem Baum finden Sie Einträge nach ihrem Namen, zum Beispiel einen bestimmten Layer oder ein Projekt. Folgende Optionen grenzen die Suche ein:

-   **Aa:** Groß- und Kleinschreibung beachten.
-   **W:** Nur ganze Wörter finden.
-   **Enter:** Zum nächsten Treffer springen.
-   **Umschalt + Enter:** Zum vorherigen Treffer springen.
-   **Escape:** Suche schließen oder Suchbegriff löschen.

Der Baum **②** zeigt dann nur noch die Treffer, jeweils mit den übergeordneten Einträgen, sodass Sie sehen, wo der Treffer liegt.

![Suche im Konfigurationsbaum](/konfigurator-images/02-baumsuche.png){border=4}

*Die Suche filtert den Baum und zeigt die Anzahl der gefundenen Einträge.*

## Pfadnavigation ::

Die Pfadnavigation über dem Editor zeigt den Weg vom Anfang der Konfiguration bis zum geöffneten Eintrag, etwa *App › projects › Stadtplan › map*. Mit einem Klick auf einen der Einträge im Pfad springen Sie direkt eine oder mehrere Ebenen zurück.

Die Pfeile für Zurück und Vorwärts blättern durch die zuletzt im Konfigurator geöffneten Einträge. Den Verlauf Ihres Browsers verändern sie nicht.

## Leitfaden

Der Leitfaden erklärt das Feld oder den Bereich, in dem Sie sich gerade befinden. Je nach Einstellung finden Sie dort:

-   eine Beschreibung, wofür die Einstellung dient,
-   die Angabe, ob das Feld ausgefüllt werden muss,
-   Hinweise, ob ein Wert optional ist oder bereits gesetzt wurde,
-   die Bezeichnung der Einstellung in der technischen Dokumentation der GBD WebSuite,
-   einen Link zu dieser Dokumentation.

## Leitfaden positionieren ::

Sie können den Leitfaden an verschiedenen Stellen anzeigen:

-   links unter dem Konfigurationsbaum,
-   rechts als eigene Spalte **①**,
-   eingeklappt, wenn Sie mehr Platz für den Editor brauchen.

Breite oder Höhe ändern Sie, indem Sie die Trennlinie verschieben. Mit dem Symbol **②** wechseln Sie die Position. Der Konfigurator merkt sich Position, Größe und ob der Leitfaden auf- oder zugeklappt ist, in Ihrem Browser.

![Rechts angedockter Leitfaden](/konfigurator-images/03-leitfaden-rechts.png){border=4}

*Der Leitfaden kann rechts als eigenes Panel angezeigt werden.*

## Werte und Felder bearbeiten

Für jede Einstellung zeigt der Konfigurator ein passendes Eingabefeld: ein Textfeld für Namen, ein Kontrollkästchen für Ein/Aus-Schalter, eine Karte für Koordinaten. Felder, die ausgefüllt werden müssen, sind mit einem Stern markiert. Jede Eingabe wird sofort als Änderung in Ihrem Entwurf vermerkt, aber noch nicht übertragen.

### Text und Boolesche Werte ::
**Text und Zahlen**

Texte wie Titel oder Namen tragen Sie direkt in das Eingabefeld ein. Bei Zahlenfeldern ist festgelegt, ob nur ganze Zahlen oder auch Dezimalzahlen erlaubt sind. Unzulässige Eingaben werden markiert und verhindern das Übertragen, bis sie korrigiert sind.

**Boolesche Werte**

Einstellungen, die nur ein- oder ausgeschaltet werden können (Ja/Nein-Werte), erscheinen als Kontrollkästchen oder Schalter. Ein Haken bedeutet: eingeschaltet.

![Textfeld und boolesche Werte](/konfigurator-images/12-text-und-boolesch.png){width=560px}

*Beispiel für ein Textfeld und mehrere aktivierte Ja/Nein-Optionen aus einer Serverkonfiguration.*

### Variantentypen ::

Manche Einstellungen gibt es in mehreren Ausprägungen, die jeweils eigene Angaben brauchen. Mit der Auswahl **Typ** legen Sie fest, welche Ausprägung gilt. Davon hängt ab, welche weiteren Felder erscheinen. Im Beispiel unten ist als Symbolform ein Kreis (*circle*) gewählt; dafür müssen Farbe und Radius angegeben werden. Wechseln Sie den Typ, können neue Pflichtfelder hinzukommen und bisherige Felder verschwinden.

%info
**Achtung:** Prüfen Sie nach einem Typwechsel alle neu erschienenen Pflichtfelder und die Liste der Änderungen, bevor Sie übertragen.
%end

![Typauswahl eines Variantentyps](/konfigurator-images/14-variantentyp.png){width=520px}

*Die Typauswahl bestimmt, welche zugehörigen Felder eingeblendet und gegebenenfalls verpflichtend werden.*

### Zeitdauer ::

Zeitangaben, etwa wie lange eine Anmeldung gültig bleibt oder wie lange Daten im Cache vorgehalten werden, geben Sie getrennt nach Tagen, Stunden, Minuten und Sekunden ein. Der Konfigurator rechnet sie in das Format um, das die GBD WebSuite erwartet.

![Eingabefelder für eine Zeitdauer](/konfigurator-images/13-zeitdauer-standardwert.png){width=520px}

*Eine Zeitdauer wird auf mehrere Eingabefelder verteilt; das Rücksetzsymbol stellt den voreingestellten Wert wieder her.*

### Koordinatenreferenzsystem ::

Das Koordinatenreferenzsystem (KBS, englisch CRS) geben Sie als EPSG-Code ein, etwa `EPSG:25832` für ETRS89 / UTM Zone 32N, oder wählen es aus einer Liste häufig verwendeter Systeme. Wenn Sie das System ändern, passen die bisherigen Koordinaten für Kartenmittelpunkt und Ausdehnung in der Regel nicht mehr. Prüfen Sie deshalb beide Werte und die Kartenvorschau.

![Auswahl eines Koordinatenreferenzsystems](/konfigurator-images/15-koordinatenreferenzsystem.png){width=520px}

*Der EPSG-Code kann direkt eingegeben oder aus häufig verwendeten Koordinatenreferenzsystemen ausgewählt werden.*

### Standardwert wiederherstellen ::

Viele Einstellungen haben einen voreingestellten Wert. Wenn Sie ihn geändert haben, stellt das Rücksetzsymbol neben dem Feld den Standard wieder her. Auch das Zurücksetzen ist eine Änderung, die im Entwurf erscheint und übertragen werden muss.

### Auswahlfelder ::

Wenn für eine Einstellung nur bestimmte Werte zulässig sind, erscheint eine Auswahlliste. Tippfehler sind damit ausgeschlossen.

### Kartenmittelpunkt ::

Der Kartenmittelpunkt bestimmt, auf welche Stelle die Karte beim Öffnen zentriert ist. Er besteht aus einer X- und einer Y-Koordinate im eingestellten Koordinatenreferenzsystem. Sie können die Werte eintippen oder, sofern die Kartenvorschau angezeigt wird, in die Karte klicken bzw. den Marker verschieben.

![X- und Y-Koordinate des Kartenmittelpunkts](/konfigurator-images/16-kartenmittelpunkt.png){width=340px}

*Direkte Eingabe der X- und Y-Koordinate des Kartenmittelpunkts.*

### Räumliche Ausdehnung ::

Die räumliche Ausdehnung (Extent) ist das Rechteck, auf das eine Karte oder ein Layer beschränkt ist. Sie wird durch die kleinsten und größten X- und Y-Koordinaten beschrieben, also durch die linke untere und die rechte obere Ecke.

![Koordinaten einer räumlichen Ausdehnung](/konfigurator-images/17-raeumliche-ausdehnung.png){width=340px}

*Der Extent wird durch minimale und maximale X- und Y-Werte begrenzt.*

Sie haben zwei Möglichkeiten, die Ausdehnung festzulegen:

1.  Die Koordinaten direkt in die vier Felder eintragen.
2.  Oder **Extent bearbeiten** wählen,
3.  ein Rechteck in der Karte aufziehen
4.  und die übernommenen Koordinaten bei Bedarf von Hand nachbessern.

Bleibt ein optionaler Extent leer, gilt keine eigene Begrenzung. Dann wird die Ausdehnung der übergeordneten Ebene verwendet, zum Beispiel die der Karte für einen Layer, oder ein von der GBD WebSuite vorgegebener Bereich.

### Kontrolle räumlicher Änderungen ::

Nach Änderungen an Koordinaten sollten Sie mindestens Folgendes prüfen:

-   Passen die Koordinaten zum eingestellten Koordinatenreferenzsystem? (Meterwerte bei UTM, Gradwerte bei WGS 84)
-   Liegt der Marker dort, wo die Karte zentriert sein soll?
-   Umfasst das Rechteck das gewünschte Gebiet?
-   Sind die Minimalwerte jeweils kleiner als die Maximalwerte?
-   Zeigt die Kartenvorschau weiterhin ein sinnvolles Bild?

### Einfache Listen ::

Manche Einstellungen bestehen aus einer Liste einfacher Einträge, etwa mehrere Texte oder Zahlen. Jeder Eintrag steht in einer eigenen Zeile.

![Einfache Liste mit skalaren Werten](/konfigurator-images/19-einfache-liste.png){width=520px}

*Einträge können ergänzt, über den Ziehgriff neu sortiert oder über das Kreuz entfernt werden.*

### Komplexe Listen ::

Andere Listen enthalten Einträge, die selbst wieder eigene Einstellungen haben. Ein typisches Beispiel ist die Liste der Layer einer Karte: Jeder Layer hat Titel, Datenquelle, Darstellung und weitere Angaben. Mit einem Klick öffnen Sie einen Eintrag und bearbeiten seine Einstellungen.

![Komplexe Liste mit Kartenlayern](/konfigurator-images/20-komplexe-liste.png){width=560px}

*Beispiel einer komplexen Liste: Jeder Layer ist ein eigenes Objekt und kann geöffnet, verschoben oder entfernt werden.*

In solchen Listen können Sie:

-   neue Einträge hinzufügen,
-   Einträge öffnen,
-   Einträge über den Ziehgriff verschieben,
-   Einträge über das Kreuz entfernen.

%info
**Achtung:** Die Reihenfolge hat oft eine Bedeutung. Bei Layern bestimmt sie zum Beispiel die Anordnung im Layerbaum, bei Zugriffsregeln, welche Regel zuerst greift.
%end

### Wörterbücher ::

Ein Wörterbuch ist eine Liste von Paaren aus Name und Wert, etwa Optionen für ein Bildformat wie `quality` = `90`. Jeder Name darf nur einmal vorkommen.

![Wörterbuch mit Schlüssel-Wert-Paaren](/konfigurator-images/21-woerterbuch.png){width=520px}

*Namen und zugehörige Werte werden zeilenweise erfasst und können neu angeordnet oder entfernt werden.*

### Freie strukturierte Werte ::

Einige wenige Felder lassen Angaben in freier Form zu. Sie werden direkt im JSON-Format eingegeben, dem Textformat, in dem die GBD WebSuite ihre Konfiguration speichert. Der Konfigurator prüft auch diese Eingaben auf Gültigkeit.

![Freier strukturierter JSON-Wert](/konfigurator-images/18-freier-json-wert.png){width=520px}

*Freie strukturierte Werte werden direkt als JSON bearbeitet und anschließend geprüft.*

## Projekte verwalten

Im Bereich **projects** stehen die Projekte Ihrer GBD WebSuite. Jedes Projekt ist eine eigene WebGIS-Anwendung mit eigener Karte, eigenen Layern und eigenen Einstellungen. **Die Projektliste ①** zeigt alle Projekte in ihrer aktuellen Reihenfolge. Mit **den Ziehgriffen ②** ändern Sie die Reihenfolge. Mit **den Symbolen ③** speichern Sie ein Projekt als Vorlage oder entfernen es. Jedes Projekt hat eine eindeutige Kennung, die UID.

Die Schaltfläche **Projekt erstellen ④** öffnet ein Menü mit den Möglichkeiten, ein neues Projekt anzulegen.

![Projektübersicht](/konfigurator-images/08-projekte.png){border=4}

*Die Projektübersicht zeigt alle vorhandenen Projekte sowie ihre Reihenfolge.*

### Projekt erstellen ::

Über **Projekt erstellen ①** gibt es je nach Situation folgende Möglichkeiten **②**:

-   **Leeres Projekt erstellen:** Legt ein neues Projekt ohne Inhalte an.
-   **Vorhandenes Projekt kopieren:** Übernimmt ein bestehendes Projekt mit allen Einstellungen als Ausgangspunkt.
-   **Projekt einfügen:** Legt ein Projekt aus einer Projektbeschreibung im JSON-Format an, zum Beispiel aus einer anderen Installation.
-   **Aus Vorlage erstellen:** Verwendet ein Projekt, das Sie vorher als Vorlage markiert haben.

![Menü zum Erstellen eines Projekts](/konfigurator-images/09-projekt-erstellen.png){border=4}

*Ein Projekt kann leer, als Kopie, aus JSON oder aus einer Vorlage erstellt werden.*

### Projektvorlage verwenden ::

Mit dem Lesezeichensymbol markieren Sie ein Projekt als Vorlage, etwa ein Musterprojekt mit Ihren üblichen Hintergrundkarten. Diese Markierung wird nur in Ihrem Browser gespeichert. Jedes aus der Vorlage erzeugte Projekt erscheint zunächst als nicht übertragene Änderung und erhält beim Übertragen mit **Anwenden** vom GBD WebSuite Server eine eigene UID.

### Projekt aus JSON einfügen ::

Zum Einfügen kopieren Sie die vollständige Projektbeschreibung im JSON-Format in das Eingabefeld des Dialogs.

So gehen Sie vor:

1.  Die vollständige Projektbeschreibung in das Eingabefeld einfügen.
2.  Gemeldete Formatfehler korrigieren.
3.  Gemeldete doppelte UIDs ändern, damit keine Kennung zweimal vorkommt.
4.  **Projekt hinzufügen** wählen.
5.  Das neue Projekt im Konfigurationsbaum öffnen.
6.  Im Änderungsbereich prüfen, was hinzugekommen ist.

Hat das Projekt keine UID, erzeugt der GBD WebSuite Server beim Übertragen mit **Anwenden** eine neue. Doppelte UIDs bei Einträgen innerhalb des Projekts, zum Beispiel bei Layern, müssen Sie dagegen selbst ändern.

## Lokale Änderungen prüfen

Jede Bearbeitung wird zunächst nur in Ihrem Browser als Änderung festgehalten. **Der Änderungsbereich ①** zeigt alle Unterschiede zwischen Ihrem Entwurf und dem Stand auf dem Server. **Die Anzeige ③** in der Kopfzeile nennt die Zahl der Änderungen; ein Klick darauf öffnet den Änderungsbereich.

![Nicht veröffentlichte Änderungen](/konfigurator-images/10-lokale-aenderungen.png){border=4}

*Der Änderungsbereich stellt bisherigen und neuen Wert gegenüber.*

**Jeder Änderungseintrag ②** zeigt den bisherigen und den neuen Wert und bietet die Schaltflächen **Öffnen** und **Verwerfen**. Je nach Art der Änderung erfahren Sie außerdem:

-   wo in der Konfiguration die Änderung liegt,
-   welche Art von Änderung es ist,
-   den bisherigen Wert,
-   den neuen Wert,
-   ob ein Eintrag neu angelegt oder gelöscht wurde.

## Änderung öffnen ::

**Öffnen** springt direkt zu der Stelle, an der die Änderung vorgenommen wurde. So sehen Sie sie im Zusammenhang mit den übrigen Einstellungen.

## Einzelne Änderung verwerfen ::

**Verwerfen** nimmt nur diese eine Änderung zurück. Es gilt wieder der Wert vom Server.

## Alle Änderungen verwerfen ::

Wenn Sie alle Änderungen verwerfen, wird Ihr gesamter Entwurf gelöscht. Auf dem Server ändert sich nichts.

## Änderungen anwenden oder verwerfen

Die Schaltfläche **Aktualisieren** öffnet eine Rückfrage. Erst wenn Sie dort **Anwenden** bestätigen, wird Ihr Entwurf an den Server übertragen und die GBD WebSuite arbeitet mit der neuen Konfiguration.

Prüfen Sie vor dem Übertragen:

-   die Anzahl der Änderungen,
-   neu angelegte Einträge,
-   gelöschte Einträge,
-   geänderte UIDs,
-   geänderte Zugriffsregeln,
-   noch nicht ausgefüllte Pflichtfelder,
-   Koordinaten und Kartenvorschau.

## Fehler beim Anwenden ::

Lehnt der Server die Übertragung ab, bleibt Ihr Entwurf vollständig erhalten. Lesen Sie die Fehlermeldung ganz, bevor Sie weiterarbeiten. Sie nennt in der Regel die Stelle, an der das Problem liegt.

So gehen Sie nach einem Fehler vor:

1.  Fehlermeldung lesen.
2.  Herausfinden, welcher Eintrag oder Wert betroffen ist.
3.  Eingabe, UID oder Zugriffsregel korrigieren.
4.  Die Liste der Änderungen noch einmal durchsehen.
5.  Erneut übertragen.

## Vollständige JSON Konfiguration

Die GBD WebSuite speichert ihre Konfiguration als Text im JSON-Format. **Die JSON-Ansicht ①** zeigt diesen Text vollständig und einheitlich formatiert.

**Der Hinweis ②** *„Nur lesen“* bedeutet, dass Sie in dieser Ansicht nichts verändern können. Sie können den Text aber durchsuchen und kopieren, etwa um einen Wert nachzuschlagen, einen Fehler einzugrenzen oder den aktuellen Stand an den Support weiterzugeben.

![Vollständige JSON Konfiguration](/konfigurator-images/06-json-konfiguration.png){border=4}

*Im Nur-Lesen-Modus kann der vollständige Stand kontrolliert und kopiert werden.*

## Bearbeitungsmodus ::

Mit der Schaltfläche **③** *„Bearbeitung aktivieren“* können Sie den JSON-Text direkt bearbeiten. Das richtet sich an Anwender, die mit dem Format vertraut sind, etwa um viele gleichartige Werte auf einmal zu ändern.

**Übernehmen** übernimmt den bearbeiteten Text in Ihren Entwurf. Auf den Server gelangt er erst über **Aktualisieren** und **Anwenden**.

Der Konfigurator lehnt den Text unter anderem ab, wenn:

-   das JSON-Format verletzt ist, etwa durch ein fehlendes Komma oder eine fehlende Klammer,
-   eine UID mehrfach vorkommt,
-   ein Wert die falsche Art hat, zum Beispiel Text statt einer Zahl,
-   Einträge so verschachtelt sind, dass sie keiner bekannten Einstellung zugeordnet werden können,
-   Pflichtangaben fehlen.

%info
**Achtung:** Im JSON-Bearbeitungsmodus lassen sich mit wenigen Zeichen viele Einstellungen gleichzeitig ändern. Nutzen Sie ihn nur, wenn Sie wissen, welche Teile der Konfiguration Sie damit verändern.
%end

## Berechtigungen

Die GBD WebSuite regelt über Rollen, wer ein Projekt sehen, einen Layer abfragen oder Einstellungen ändern darf. **Die Berechtigungsübersicht ①** zeigt alle Einträge der Konfiguration in ihrer Baumstruktur. **Die farbigen Kennzeichnungen ②** zeigen für jeden Eintrag, welche Rollen ihn tatsächlich lesen, bearbeiten, neu anlegen und löschen dürfen. Diese Angaben berechnet der Server aus allen geltenden Regeln, einschließlich der von übergeordneten Einträgen geerbten.

![Berechtigungsübersicht ohne lokale Änderungen](/konfigurator-images/04-berechtigungen.png){border=4}

*Wenn keine lokalen Änderungen vorliegen, werden die tatsächlich geltenden Rollen angezeigt.*

%warn
**Wichtig bei nicht übertragenen Änderungen:** Solange Ihr Entwurf Änderungen enthält, kann der Konfigurator die tatsächlich geltenden Berechtigungen nicht zuverlässig anzeigen. Der Server hat sie für den bisherigen Stand berechnet, Ihr Entwurf weicht davon aber bereits ab. Deshalb erscheint **der Hinweis ①** *„Änderungen übernehmen oder verwerfen.“*, und die vom Server berechneten Rollen werden ausgeblendet.
%end

![Berechtigungsübersicht bei nicht veröffentlichten Änderungen](/konfigurator-images/11-berechtigungen-gesperrt.png){border=4}

*Bei lokalen Änderungen sind die tatsächlich geltenden Rollen vorübergehend nicht sichtbar.*

Die Berechtigungen werden wieder angezeigt, sobald eine der beiden Bedingungen erfüllt ist:

1.  Sie verwerfen alle Änderungen. Dann entspricht die Anzeige wieder dem Stand auf dem Server.
2.  Sie übertragen die Änderungen mit **der Schaltfläche ③** *„Aktualisieren“* und anschließend **Anwenden**. Der Konfigurator wartet dann, bis der Server die Berechtigungen neu berechnet hat.

In der Zwischenzeit sehen Sie weiterhin die eingetragenen Zugriffsregeln **②**, etwa `allow all` oder `allow admin-1`, und können sie bearbeiten. Diese Regeln zeigen aber nur, was an genau dieser Stelle eingetragen ist, nicht, was am Ende tatsächlich gilt. Vor allem geerbte Rechte können Sie erst nach dem Verwerfen oder Übertragen wieder vollständig beurteilen.

### ACL Regeln bearbeiten ::

Zugriffsregeln (ACL, Access Control List) legen fest, welche Rollen etwas dürfen und welche nicht. Eine Regel besteht aus einer Entscheidung (erlauben oder verweigern) und einer Rolle. Mehrere Regeln werden der Reihe nach von oben nach unten geprüft.

![Editor für ACL-Regeln](/konfigurator-images/22-acl-regeln.png){width=520px}

*Jede Zugriffsregel enthält eine Entscheidung, eine Rolle sowie Schaltflächen zum Verschieben und Entfernen.*

So legen Sie eine Regel an:

1.  **Erlauben** oder **Verweigern** wählen.
2.  Die Rolle eintragen oder auswählen, zum Beispiel `all` für alle Besucher oder eine Benutzergruppe.
3.  Die Regel an die richtige Stelle in der Reihenfolge schieben.
4.  Bei Bedarf weitere Regeln hinzufügen oder nicht mehr benötigte entfernen.

%warn
**Wichtig:** Die erste Regel, die auf einen Benutzer zutrifft, entscheidet. Steht zum Beispiel `allow all` vor `deny guest`, dürfen auch Gäste zugreifen. Achten Sie deshalb genau auf die Reihenfolge.
%end

Die Rolle `admin` hat immer alle Rechte, unabhängig von den eingetragenen Regeln. Für alle anderen Rollen prüft der Konfigurator beim Übertragen nicht, ob Sie sich oder anderen versehentlich den Zugang entziehen.

### Berechtigungsübersicht ::

Die Berechtigungsübersicht zeigt für alle Einträge der Konfiguration, welche Rechte tatsächlich gelten.

Sie beantwortet zum Beispiel diese Fragen:

-   Welche Rollen dürfen ein Projekt oder einen Layer sehen?
-   Welche Rollen dürfen Einstellungen ändern?
-   Welche Rollen dürfen darunter neue Einträge anlegen?
-   Welche Rollen dürfen einen Eintrag löschen?
-   Welche Rechte stammen von einem übergeordneten Eintrag?
-   An welcher Stelle verliert eine Rolle ihren Zugriff?

Wenn Sie Zugriffsregeln geändert und übertragen haben, öffnen Sie die Berechtigungsübersicht erneut und prüfen Sie dort das Ergebnis.

## Darstellung und responsive Nutzung

## Heller und dunkler Modus ::

Mit dem Sonnen- bzw. Mondsymbol **①** wechseln Sie zwischen heller und dunkler Darstellung.

![GBD Konfigurator im dunklen Modus](/konfigurator-images/07-dunkler-modus.png){border=4}

*Im dunklen Modus werden Oberfläche, Statusfarben und Eingabefelder gemeinsam angepasst.*

Die Einstellung wird nur in Ihrem Browser gespeichert und hat keinen Einfluss auf die Konfiguration der GBD WebSuite.

## Schmale Bildschirme ::

Auf schmalen Bildschirmen, etwa auf einem Tablet, legt sich der Konfigurationsbaum über den Editor. Nachdem Sie einen Eintrag gewählt haben, kann er sich automatisch schließen, damit der Editor mehr Platz hat.

Auch der Leitfaden passt sich der Breite an. Für umfangreiche Tabellen, die JSON-Ansicht und die Berechtigungsübersicht ist ein größerer Bildschirm trotzdem empfehlenswert.

### Tastaturbedienung ::

| Taste | Funktion |
| --- | --- |
| `Enter` | Zum nächsten Suchtreffer springen |
| `Umschalt + Enter` | Zum vorherigen Suchtreffer springen |
| `Escape` | Suche oder geöffneten Dialog schließen |
| `Pfeil nach oben` / `Pfeil nach unten` | Zur darüber- oder darunterliegenden Zeile wechseln |
| `Leertaste` | Ausgewählten Eintrag im Baum auf- oder zuklappen |

## Fehler einordnen
### Konfiguration kann nicht geladen werden ::

Mögliche Ursachen:

-   Der Server ist nicht erreichbar.
-   Ihre Anmeldung ist abgelaufen.
-   Ihnen fehlt das Recht, die Konfiguration zu lesen.
-   Die Konfiguration auf dem Server enthält Fehler.

Prüfen Sie Netzwerkverbindung, Anmeldung und Berechtigungen und laden Sie die Seite anschließend neu.

### Ungültiges JSON ::

Der JSON-Text enthält einen Formatfehler. Häufige Ursachen sind fehlende Kommas, fehlende Anführungszeichen und nicht geschlossene Klammern.

### Doppelte UID ::

Jeder Eintrag der Konfiguration braucht eine eigene, eindeutige Kennung (UID). Öffnen Sie die gemeldeten Einträge und geben Sie der Kopie oder dem neu eingefügten Eintrag eine neue UID.

### Pfad kann nicht aufgelöst werden ::

Der Eintrag, den Sie öffnen wollten, existiert an dieser Stelle nicht mehr. Öffnen Sie einen übergeordneten Eintrag im Baum und prüfen Sie, ob der gesuchte Eintrag gelöscht, verschoben oder durch einen Typwechsel ersetzt wurde.

### Kartenvorschau fehlt ::

Prüfen Sie Koordinatenreferenzsystem, Koordinaten, die Datenquelle der Karte und Ihre Berechtigungen. Unpassende Koordinaten oder eine nicht erreichbare Datenquelle verhindern die Vorschau.

### Berechtigungsanzeige ist veraltet ::

Übertragen oder verwerfen Sie Ihre Änderungen und öffnen Sie die Berechtigungsübersicht anschließend erneut.

## Empfohlener Arbeitsablauf

1.  Das gewünschte Thema oder Projekt öffnen.
2.  Im Leitfaden nachlesen, wofür die Einstellung dient, und den aktuellen Wert ansehen.
3.  Den Wert ändern.
4.  Prüfen, ob dadurch weitere Felder oder Pflichtangaben hinzugekommen sind.
5.  Den Entwurf im Änderungsbereich durchsehen.
6.  Geänderte UIDs, gelöschte Einträge und geänderte Zugriffsregeln besonders genau prüfen.
7.  Die Änderungen über **Aktualisieren** und **Anwenden** übertragen.
8.  Die Rückmeldung des Servers abwarten.
9.  Den geänderten Bereich erneut öffnen.
10.  Das Ergebnis prüfen, nach Änderungen an Zugriffsregeln auch in der Berechtigungsübersicht.

Wer diese Reihenfolge einhält, vermeidet versehentliche Änderungen und behält auch bei umfangreichen Arbeiten den Überblick.

## Feldtypen und Symbole

Welche Feldtypen bei Ihnen erscheinen, hängt von der Version der GBD WebSuite ab. In der Tabelle stehen die deutschen Bezeichnungen und die englischen Fachbegriffe, wie sie im Leitfaden und in der technischen Dokumentation verwendet werden.

| Gruppe | Beispiele |
| --- | --- |
| Einfache Werte | Text, Ganzzahl, Dezimalzahl, Ja/Nein-Wert (Boolean), beliebiger Wert (Any) |
| Auswahl | Auswahlliste (Enum), Variantentyp (Variant) |
| Zeit | Zeitdauer (Duration) |
| Raumbezug | Koordinatenreferenzsystem (CRS), Punkt (Point), Ausdehnung (Extent) |
| Werte mit Einheit | Wert mit Maßeinheit (UOM Value), Größe (Size), Ausdehnung (Extent) |
| Zugriff | Zugriffsregeln (ACL) |
| Sammlungen | Liste (List), Wörterbuch (Dict), Menge (Set) |
| Eigene Bereiche | Projekte (Projects), Karte (Map), Berechtigungen (Permissions) |

Häufige Symbole und Kennzeichnungen:

| Anzeige | Bedeutung |
| --- | --- |
| Stern | Pflichtfeld |
| Hervorgehobene Zeile | Gerade ausgewählter Eintrag |
| Häkchen | Vorgang erfolgreich abgeschlossen |
| Ausrufezeichen oder Fehlermeldung | Fehler des Servers, der Verbindung oder bei der Prüfung einer Eingabe |
| Geschweifte Klammern | JSON-Ansicht |
| Mond oder Sonne | Wechsel zwischen heller und dunkler Darstellung |
| Schloss | Berechtigungsübersicht |
| Ziehgriff | Eintrag verschieben oder Bereich vergrößern/verkleinern |

Die Kennzeichnung **Unveröffentlicht** bedeutet, dass Sie einen Wert geändert, aber noch nicht über **Aktualisieren** und **Anwenden** an den Server übertragen haben.

## Begriffe

### Schema ::

Von der GBD WebSuite mitgelieferte Beschreibung aller Einstellungen, die es gibt: welche Felder vorhanden sind, welche ausgefüllt werden müssen, welche Werte erlaubt und welche voreingestellt sind. Aus dem Schema baut der Konfigurator seine Eingabemasken auf.

### Konfiguration ::

Gesamtheit aller Einstellungen einer GBD WebSuite Installation, von globalen Angaben wie Datenbankverbindungen bis zu einzelnen Projekten und Layern. Sie wird als JSON-Text gespeichert.

### UID ::

Eindeutige Kennung eines Eintrags in der Konfiguration, vergleichbar mit einem Primärschlüssel in einer Datenbanktabelle. Jede UID darf nur einmal vorkommen.

### ACL ::

Access Control List, auf Deutsch Zugriffsliste: eine geordnete Folge von Regeln, die festlegen, welche Rollen etwas dürfen und welche nicht.

### CRS ::

Coordinate Reference System, auf Deutsch Koordinatenreferenzsystem: legt fest, wie Koordinaten zu verstehen sind. Es wird in der Regel als EPSG-Code angegeben, etwa `EPSG:25832`.

### Extent ::

Räumliche Ausdehnung: ein Rechteck, das durch die kleinste und größte X- und Y-Koordinate beschrieben wird.

### Lokaler Entwurf ::

Alle Änderungen, die Sie im Browser vorgenommen, aber noch nicht an den Server übertragen haben.

### Anwenden ::

Letzter Schritt, mit dem Sie Ihren geprüften Entwurf an den Server übertragen. Er muss in einer Rückfrage ausdrücklich bestätigt werden.

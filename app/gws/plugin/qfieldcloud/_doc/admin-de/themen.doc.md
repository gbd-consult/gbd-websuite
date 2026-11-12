# QField / QFieldCloud :/admin-de/themen/fachmodule/qfieldcloud

[QField](https://qfield.org) ist eine mobile Anwendung auf Basis von QGIS-Projektdateien, mit der man Geodaten im Außendienst erfasst. Sollen Daten mit QField offline bearbeitet werden, so sind eine entsprechende Bereitstellung der Daten und ein Synchronisieren notwendig.

Die GBD WebSuite bildet dafür die QFieldCloud-Schnittstelle nach, sodass QField unmittelbar mit ihr synchronisiert und kein zusätzlicher Cloud-Dienst nötig ist.

Für die App ist der Ablauf derselbe wie bei der QFieldCloud: anmelden, ein Datenpaket herunterladen, offline damit arbeiten, die Änderungen später zurückschicken. Die WebSuite stellt bei jeder Anforderung aus einem QGIS-Projekt und den Datenbanktabellen ein neues Datenpaket für QField bereit. Beim Synchronisieren schreibt sie die zurückgemeldeten Änderungen über das Datenmodell wieder in dieselben Tabellen zurück.

Die Einrichtung besteht aus zwei Teilen, die zusammenpassen müssen: Im QGIS-Projekt legen Sie mit der Erweiterung QFieldSync fest, was mit welchem Layer geschehen soll. In der WebSuite konfigurieren Sie eine Aktion, die dieses Projekt anbindet und die Berechtigungen setzt.

Die Konfiguration des Fachmoduls QField / QFieldCloud setzt voraus, dass Sie sich parallel mit der Konfiguration von QGIS-Projekten für QField befassen. Der folgende Abschnitt fasst die Grundlagen kurz zusammen und verweist auf die QField-Dokumentation. Alle weiteren Abschnitte beschreiben die Umsetzung in der WebSuite.

## Grundlagen: QGIS-Projekte für QField

QField zeigt ein QGIS-Projekt auf dem Mobilgerät an, und zwar so, wie es in QGIS konfiguriert ist. Fast alles, was die Feldkräfte später sehen und bedienen, legen Sie deshalb in QGIS fest:

- **Layer und Darstellung:** Symbolisierung, Beschriftung, Kartenthemen und Maßstabsbereiche übernimmt QField unverändert.
- **Attributformulare:** In den Layereigenschaften unter *Attributformular* legen Sie fest, wie Felder in der App erfasst werden: Bedienelemente wie Wertzuordnung, Datum oder *Anhang*, Vorgabewerte, Beschränkungen und bedingte Sichtbarkeit. Beziehungen zwischen Layern stellt QField über die Bedienelemente *Beziehungsreferenz* und *Beziehungseditor* dar.
- **QFieldSync:** Die QGIS-Erweiterung QFieldSync ergänzt das Projekt um die Angaben, die nur für QField gelten. Über das Menü *Configure current project* / *QFieldCloud packing* stellen Sie je Layer ein, wie er beim Packen behandelt werden soll. Dort legen Sie auch Hintergrundkarte, Bearbeitungsgebiet und Verzeichnisse für Anhänge fest.
- **Paket und Synchronisierung:** Aus dem Projekt entsteht ein Paket, das die App herunterlädt. Mit diesem Paket wird offline gearbeitet. Die Änderungen gehen später per Synchronisierung zurück.

## Besonderheiten der Bereitstellung durch die WebSuite

| Thema | QFieldCloud | GBD WebSuite |
|---|---|---|
| Projekt bereitstellen | Projekt wird aus QGIS in die Cloud hochgeladen | Die WebSuite liest das QGIS-Projekt selbst. Angebunden wird es über eine [Aktion](/admin-de/themen/fachmodule/qfieldcloud/die-aktion-einrichten) |
| Benutzer und Rechte | Konten, Organisationen und Teams in QFieldCloud | Benutzerkonten und Rollen der WebSuite, siehe [](/admin-de/themen/fachmodule/qfieldcloud/berechtigungen) |
| Bearbeitbare Daten | verschiedene Formate | PostgreSQL/PostGIS-Tabellen |
| Hintergrundkarte, Bearbeitungsgebiet | müssen manuell erstellt werden | werden nach Konfiguration in QFieldSync von der WebSuite erstellt, siehe [](/admin-de/themen/fachmodule/qfieldcloud/das-qgis-projekt-mit-qfieldsync-vorbereiten/hintergrundkarte) |
| Befugnisse | gelten nur auf dem Gerät | werden auch beim Zurückschicken geprüft und mit den Rechten der Datenmodelle verrechnet |
| Anhänge (Fotos) | Dateien liegen als Dateien vor | Dateien liegen auf dem Server oder in der Datenbank, siehe [](/admin-de/themen/fachmodule/qfieldcloud/arbeiten-mit-dateien) |

## Voraussetzungen

Für QGIS-Projekte, die die WebSuite für QField bereitstellt, gilt:

- Das QGIS-Projekt liegt als Datei oder in einer PostGIS-Datenbank vor. Sollen Anhänge oder weitere Dateien mit ins Paket, müssen diese als Dateien vorliegen. Nur dann lassen sich die in QFieldSync relativ angegebenen Verzeichnisse auflösen.
- Alle Layer, die im Feld bearbeitet werden sollen, müssen aus PostgreSQL/PostGIS stammen. Andere Quellen lassen sich nicht offline bearbeiten. Die WebSuite entfernt solche Layer aus dem Paket und schreibt eine Warnung ins Protokoll.
- Konfiguration der Layer mit der QGIS-Erweiterung QFieldSync.
- Benutzerkonten für die Feldkräfte. Die App meldet sich mit denselben Zugangsdaten an wie ein Browser-Nutzer. Geprüft werden sie über die konfigurierten Provider der [](/admin-de/themen/zugriff/authentifizierung). Dazu kommt eine dauerhafte Sitzungsverwaltung.

## Das QGIS-Projekt mit QFieldSync vorbereiten

### Grundeinstellungen

Sie bereiten das Projekt über Projekt- und Layereigenschaften und mit QFieldSync im Bereich *QFieldCloud packing* so vor, wie es in den [Grundlagen](/admin-de/themen/fachmodule/qfieldcloud/grundlagen-qgis-projekte-fuer-qfield) kurz umrissen ist. Weiter Information finden Sie in der QField Dokumentation. 

QFieldSync speichert die vorgenommenen Einstellungen als Eigenschaften im Projekt. Die WebSuite liest diese Eigenschaften aus und richtet sich danach. Sie konfigurieren das Verhalten der Layer also in QGIS, nicht in der WebSuite.

Ausgewertet werden nur die Angaben, die das Paket betreffen: die Aktion je Layer, die Hintergrundkarte, das Bearbeitungsgebiet und die mitzukopierenden Verzeichnisse. Alle übrigen QFieldSync-Einstellungen, etwa zu Geofencing, Positionsaufzeichnung oder Bildstempeln, bleiben unverändert im Projekt stehen. QField wertet sie selbst auf dem Gerät aus und sie funktionieren wie in der QField-Dokumentation beschrieben.

Je Layer entscheidet die eingestellte Aktion, was mit ihm im Paket geschieht. QFieldSync bietet je nach Layertyp mehr Aktionen an, als die WebSuite unterscheidet. Für die WebSuite zählen nur diese drei:

- **Offline bearbeiten** (*Offline Editing*) – der Layer wird bearbeitbar. Die WebSuite exportiert seine Objekte in ein GeoPackage, das mit ins Paket kommt, und stellt den Layer im mitgelieferten Projekt auf diese Datei um. Das setzt eine PostGIS-Tabelle voraus, und zwar eine echte Tabelle: Layer, die auf einer Unterabfrage oder einem `SELECT` beruhen, lassen sich nicht zuordnen und werden entfernt.
- **Entfernen** (*Remove from Project*) – der Layer wird aus dem mitgelieferten Projekt gelöscht. Gruppen, die dadurch leer werden, verschwinden ebenfalls.
- **Direkter Zugriff** (*Directly access data source*) – direkter Zugriff auf einen PostGIS-Datenbanklayer, Online-Kartendienste oder einen virtuellen Layer.

Weitere Einstellungen zur Bearbeitbarkeit eines Layers können Sie über QFieldSync oder direkt in den Layereigenschaften unter *QField* definieren. Einzelne Befugnisse verrechnet die WebSuite mit den Rechten am Datenmodell, wie unter [](/admin-de/themen/fachmodule/qfieldcloud/berechtigungen) beschrieben.

Beachten Sie in QGIS außerdem die Einstellungen unter *Projekt* > *Eigenschaften* > *Datenquellen* zum Schreibschutz in Projekten und zu weiteren Eigenschaften.

### Bearbeitungsgebiet

Legen Sie in QFieldSync ein Bearbeitungsgebiet (*Area of Interest*) fest, nutzt die WebSuite es als Ausschnitt für die Hintergrundkarte. Ist zusätzlich eingestellt, dass nur dieses Gebiet kopiert werden soll, exportiert sie auch die Objekte der bearbeitbaren Layer nur aus diesem Bereich. Ohne Bearbeitungsgebiet gilt die Ausdehnung des QGIS-Projekts. Bei einem landesweiten Datenbestand ist das entsprechend viel.

### Hintergrundkarte

Ist in QFieldSync unter *QFieldCloud packing* die Erzeugung einer Hintergrundkarte (*Base Map*) aktiv, erstellt die WebSuite ein entsprechendes Rasterbild und fügt es dem Paket hinzu. Als Vorlage dient wahlweise ein Kartenthema des Projekts oder ein einzelner Layer. Von den eingestellten Zoomstufen nimmt die WebSuite die höhere und begrenzt sie auf den Bereich 3 bis 20. Daraus ergibt sich die Auflösung des Bildes.

Das Rendern ist der mit Abstand aufwendigste Teil der Paketerstellung. Halten Sie das Gebiet klein (Bearbeitungsgebiet) und die Zoomstufe so niedrig wie vertretbar, und schalten Sie in der Aktion einen Cache ein. Das Bild darf höchstens eine Milliarde Pixel haben, etwa 31.000 x 31.000. Bei Zoomstufe 18 entspricht das einem Gebiet von rund 19 x 19 km, bei Stufe 16 von rund 75 x 75 km. Ist das Gebiet größer, bricht die Paketerstellung mit einer Fehlermeldung im Protokoll ab.

Alternativ können Sie Rasterbilder und andere relevante Grundlagendaten auf Ihrem mobilen Gerät dauerhaft zur Verwendung unter *QField/basemaps* ablegen. So können Sie auch verschiedene Hintergrundbilder verwenden. Beachten Sie dazu die Hilfe zu *shared datasets* in der QField-Dokumentation.

### Anhänge und mitgelieferte Dateien

Die in QFieldSync angegebenen Verzeichnisse für Anhänge und Daten kopiert die WebSuite unverändert ins Paket. Relative Pfade bezieht sie auf den Ort der Projektdatei. Verschachtelte Angaben werden zusammengefasst, sodass kein Verzeichnis doppelt im Paket landet.

### Aktuellen Nutzer erfassen

Möchten Sie bei der Datenerfassung den aktuell eingeloggten Erfassen in den Datensatz zufügen, so erstellen in dem Datensatz ein Textfeld und setzten Sie  `@cloud_username` als Vorgabewert ein.

## Die Aktion einrichten

Anders als bei QFieldCloud laden Sie das Projekt nicht aus QGIS hoch. Die WebSuite liest die Projektdatei direkt, die Anbindung geschieht über die folgende Konfiguration.

In der WebSuite binden Sie das vorbereitete Projekt über eine Aktion vom Typ `qfieldcloud` an. Jeder Eintrag unter `projects` ist ein Projekt, das in der App zur Auswahl steht:

```javascript
actions+ {
    type "qfieldcloud"
    access "allow all"
    projects+ {
        uid "erfassung"
        title "Baumkataster Außendienst"
        provider.path "/data/qgis/baeume.qgs"
        mapCacheLifeTime "7d"
        access "allow user, deny all"
        models+ {
            type "postgres"
            tableName "erfassung.baum"
            withAutoFields true
            isEditable true
            permissions.edit "allow feldkraft, deny all"
        }
    }
}
```

Der Abschnitt `models` ist dabei **nicht** zwingend. Findet die WebSuite zu einer bearbeitbaren Tabelle kein konfiguriertes Modell, legt sie selbst eines an, das alle Spalten übernimmt und Bearbeiten zulässt. Für einen ersten Versuch genügt daher `provider.path`. Ein eigenes Modell brauchen Sie, sobald Sie die Felder einschränken, die Bearbeitung auf bestimmte Rollen begrenzen oder Dateianhänge in der Datenbank ablegen wollen.

Alle Optionen samt Beispielen stehen auf der [Konfigurationsseite der Aktion](/admin-de/konfiguration/action/qfieldcloud).

## Zugang für die App

Die Aktion beantwortet die Anfragen der App unter einem internen Befehl. Damit die Nutzer eine gewöhnliche Adresse eintragen können, legen Sie in der Webseite eine Rewrite-Regel an, die einen Pfad Ihrer Wahl darauf abbildet.

Die QField-Anwender wählen in der App *QFieldCloud-Projekte*, tragen die Adresse aus der Rewrite-Regel als Server ein und melden sich mit ihren WebSuite-Zugangsdaten an. Sie sehen dann im Dialog die verfügbaren Projekte. Anschließend laden sie das Projekt aus der Liste herunter und übertragen ihre Änderungen später über *Synchronisieren* zurück.

Die Anmeldung selbst müssen Sie nicht einrichten. Die Aktion bringt eine eigene Authentifizierungsmethode mit und registriert sie beim Start. Sie taucht deshalb nicht unter den Methoden auf, die Sie in der Applikation auswählen. Diese Methode prüft die von der App übermittelten Zugangsdaten über die Authentifizierungs-Provider und gibt ein Token zurück, das die Kennung einer Sitzung ist. Jede weitere Anfrage der App weist dieses Token vor und verlängert damit die Sitzung. Meldet sich ein Nutzer in der App ab, wird die Sitzung gelöscht.

Ein Token gilt also so lange wie eine Sitzung. Richten Sie deshalb eine dauerhafte [Sitzungsverwaltung](/admin-de/konfiguration/authSessionManager) ein. Ohne sie gehen alle Anmeldungen bei jedem Serverneustart verloren, und die Feldkräfte stehen im ungünstigsten Moment ohne Zugang da.

## Was im Datenpaket steckt

Fordert die App ein Paket an, legt die WebSuite einen Hintergrundauftrag an und baut es zusammen:

- je bearbeitbarem Layer ein GeoPackage mit den exportierten Objekten,
- die gerenderte Hintergrundkarte als Rasterbild,
- die kopierten Anhang- und Datenverzeichnisse,
- das QGIS-Projekt selbst, umgeschrieben auf die enthaltenen Dateien.

Beim Umschreiben ersetzt die WebSuite die Datenquellen der bearbeitbaren Layer und der Hintergrundkarte durch die lokalen Dateien, entfernt die ausgeschlossenen Layer und stellt das Projekt auf relative Pfade um. Auch Beziehungen zwischen Layern und Bedienelemente vom Typ *Beziehungsreferenz* (*Relation Reference*) werden übernommen, sodass verknüpfte Objekte im Feld weiter funktionieren. Verwenden Sie UUIDs als Primärschlüssel, insbesondere bei der Arbeit mit Relationen und in Teams. Beachten Sie dazu die QField-Dokumentation.

Zwei Eigenheiten des Exports sollten Sie kennen, wenn Sie Formulare in QGIS gestalten:

- Nur Attribute der Typen Wahrheitswert, Datum, Zeit, Zeitstempel, Ganzzahl, Fließkommazahl und Text landen im GeoPackage. Felder anderer Typen fehlen in der App. Wie Dateifelder behandelt werden, beschreibt der Abschnitt [](/admin-de/themen/fachmodule/qfieldcloud/arbeiten-mit-dateien).
- Ein Feld namens `fid` benennt die WebSuite im Paket in `fid_gws` um, weil GDAL diesen Namen selbst belegt. Ausdrücke im Formular müssen in der App deshalb `fid_gws` verwenden. Auf dem Rückweg macht die WebSuite die Umbenennung wieder rückgängig.

Enthält das QGIS-Projekt die Platzhalter `{user.authToken}`, `{user.loginName}` oder `{user.displayName}`, setzt die WebSuite beim Packen die Werte des anfordernden Nutzers ein. So kann ein Layer im Feld etwa einen Dienst der WebSuite selbst ansprechen, ohne dass Zugangsdaten im Projekt stehen.

## Der Rückweg: Änderungen und Anhänge

QField schickt die im Feld vorgenommenen Änderungen als Liste von Einzeloperationen zurück – anlegen, ändern, löschen. Die WebSuite ordnet jede Operation dem Datenmodell des betroffenen Layers zu und führt sie darüber aus. Damit gelten dieselben Regeln wie beim [](/admin-de/themen/daten/editieren) im Browser: Validatoren, Wertvorgaben und Berechtigungen greifen unverändert. Alle Operationen einer Übertragung laufen gemeinsam in einer Transaktion. Schlägt eine fehl, wird nichts geschrieben.

Vorgabewerte und Beschränkungen aus dem QGIS-Attributformular prüft nur QField auf dem Gerät. Auf dem Server gelten zusätzlich die Validatoren und Wertvorgaben der WebSuite-Modelle. Was dort abgelehnt wird, lässt die ganze Übertragung scheitern. Stimmen Sie beides aufeinander ab.

Eine abgelehnte Übertragung geht nicht verloren. QField behält die Änderungen auf dem Gerät und meldet die Synchronisierung als fehlgeschlagen. Solange die Ursache besteht, etwa eine verletzte Validierung oder eine fehlende Berechtigung, scheitert jeder weitere Versuch. Auch spätere Änderungen auf diesem Gerät gelangen dann nicht auf den Server. Den Grund nennt das Protokoll der WebSuite. Ist er behoben, überträgt die nächste Synchronisierung alles. Bereits übernommene Änderungen erkennt die WebSuite wieder und schreibt sie kein zweites Mal. Verwerfen Sie lokale Änderungen in der App erst, wenn die Ursache geklärt ist, sonst gehen sie verloren.

Weil eine Übertragung eine einzige Transaktion ist, müssen alle Tabellen eines QField-Projekts über dieselbe Datenbankverbindung laufen: die Modelle unter `models` ebenso wie die Layer, die in QFieldSync zum Bearbeiten markiert sind. Verteilen sie sich auf mehrere [Datenbank-Provider](/admin-de/konfiguration/databaseProvider), lehnt die WebSuite jede Übertragung ab.

Dateianhänge kommen getrennt hinterher, wie unter [](/admin-de/themen/fachmodule/qfieldcloud/arbeiten-mit-dateien) beschrieben.

## Berechtigungen

Bei QFieldCloud steuern Organisationen, Teams und Projektrollen den Zugriff. In der WebSuite gibt es diese Begriffe nicht. Stattdessen gelten die Benutzer, Rollen und Zugriffsregeln der WebSuite. Definieren Sie wie in einer Datenbankanwendung Rollen für Ihr Team. So können Sie z. B. steuern, wer welche Projekte verwenden kann.

Es greifen drei Ebenen, und alle drei sollten Sie bewusst setzen:

- Die **Aktion** selbst legt fest, wer die Schnittstelle überhaupt ansprechen darf.
- Jeder **Projekteintrag** hat eigene Zugriffsregeln. In der App sieht ein Nutzer nur die Projekte, die er benutzen darf.
- Die **Layer** bestimmen, was bearbeitet werden darf: über die *Befugnisse* in QFieldSync, über die Rechte an den Modellen oder über beides.

Für die Bearbeitung gibt es damit zwei Wege. Sollen alle Nutzer eines Projekts dieselben Möglichkeiten haben, genügen die Befugnisse in QFieldSync. Brauchen verschiedene Nutzergruppen verschiedene Rechte, steuern Sie die Bearbeitung über die Modelle der WebSuite.

### Berechtigungen mit QFieldSync steuern

Sie konfigurieren keine eigenen Modelle, oder nur Modelle, die allen Nutzern die Bearbeitung erlauben (`permissions.edit "allow user"`). Die automatisch erzeugten Modelle sind für jeden bearbeitbar, der das Projekt benutzen darf. Wer zugreift, entscheiden dann allein die Zugriffsregeln der Aktion und des Projekteintrags.

Was bearbeitet werden darf, legen Sie je Layer in den Layereigenschaften unter *QField* > *Befugnisse* fest:

- *Erstellen von Objekt deaktivieren*
- *Löschen von Objekten deaktivieren*
- *Bearbeiten von Attributen deaktivieren*
- *Bearbeiten von Geometrien deaktivieren*

Aktivieren Sie alle vier Optionen, behandelt die WebSuite den Layer als reine Lesequelle. Er wird dann zwar ins Paket exportiert, braucht aber kein bearbeitbares Datenmodell.

Diese Einstellungen gelten für alle Nutzer gleich. QField bietet die deaktivierten Werkzeuge nicht an, und die WebSuite prüft sie beim Zurückschicken erneut: Eine Änderung, deren Befugnis im QGIS-Projekt deaktiviert ist, weist sie ab. Da alle Operationen einer Übertragung in einer Transaktion laufen, wird dann die ganze Übertragung abgelehnt.

Befugnisse, die über einen Ausdruck gesteuert werden (das Symbol neben der Option), wertet die WebSuite nicht aus. Der Ausdruck bleibt im Projekt stehen und wird nur von QField auf dem Gerät ausgewertet; beim Zurückschicken prüft die WebSuite ihn nicht.

### Berechtigungen mit der WebSuite steuern

Sie konfigurieren Modelle mit `isEditable` und vergeben die Rechte getrennt nach Rollen (`permissions.create`, `permissions.write`, `permissions.delete`). So kann etwa eine Gruppe Objekte nur ändern, eine andere auch anlegen und löschen.

Die Befugnisse in QFieldSync gelten dabei weiter. Eine Bearbeitung ist nur erlaubt, wenn das Modell sie dem Nutzer erlaubt und die zugehörige Befugnis nicht deaktiviert ist:

| Befugnis in QFieldSync | nötiges Recht am Modell |
|---|---|
| *Erstellen von Objekt deaktivieren* | `permissions.create` |
| *Löschen von Objekten deaktivieren* | `permissions.delete` |
| *Bearbeiten von Attributen deaktivieren* | `permissions.write` |
| *Bearbeiten von Geometrien deaktivieren* | `permissions.write` am Modell und am Geometriefeld |

Ohne `isEditable` am Modell, oder wenn der Layer in QGIS schreibgeschützt ist, ist jede Bearbeitung deaktiviert. Das Ergebnis berechnet die WebSuite beim Packen für den anfordernden Nutzer und setzt die Befugnisse im mitgelieferten Projekt entsprechend. Ist jede Bearbeitung deaktiviert, kommt der Layer schreibgeschützt ins Paket. Zwei Feldkräfte können daher vom selben QGIS-Projekt unterschiedliche Pakete erhalten.

Beim Zurückschicken greifen die Rechte am Modell wie beim Editieren im Browser. Was auf dem Server verbindlich ausgeschlossen sein soll, regeln Sie hier und nicht über einen Ausdruck in QFieldSync.

## Arbeiten mit Dateien

Fotos und andere Anhänge, die im Feld entstehen, legt die WebSuite in der Datenbank ab, in derselben Tabelle wie das Objekt. Auf Wunsch schickt sie vorhandene Dateien auch mit dem Paket an die App.

### Spalten in der Datenbank

Je Anhang braucht die Tabelle zwei Spalten:

- eine Textspalte für den Dateinamen, genauer den Pfad, den QField für die Datei vergibt, etwa `DCIM/foto_20261016.jpg`,
- eine Spalte vom Typ `bytea` für den Dateiinhalt.

Die Namensspalte ist ein gewöhnliches Attribut: Sie kommt ins GeoPackage, QField trägt dort den Pfad ein und schickt ihn mit den übrigen Änderungen zurück. Die Inhaltsspalte kommt nicht ins Paket, solange Sie sie nicht über `syncFiles` mit `useBlob` anfordern (siehe unten).

### Das Dateifeld im Modell

Damit die WebSuite die beiden Spalten als Anhang erkennt, braucht das Modell ein Feld vom Typ [`file`](/admin-de/konfiguration/modelField/file), das sie verbindet. Das Feld ist virtuell: Sein Name entspricht keiner Spalte, `nameColumn` und `contentColumn` verweisen auf die beiden echten Spalten. Die Namensspalte bleibt daneben als eigenes Textfeld im Modell, etwa über `withAutoFields`:

```javascript
models+ {
    type "postgres"
    tableName "edit.poi"
    withAutoFields true
    isEditable true
    fields+ {
        type "file"
        name "photo_file"
        nameColumn "photo"
        contentColumn "photo_content"
    }
}
```

Automatisch erzeugte Modelle enthalten kein Dateifeld. Für jede Tabelle mit Anhängen ist deshalb ein eigenes Modell nötig.

QField überträgt einen neuen Anhang in zwei Schritten: zuerst die Änderung am Objekt, in der nur der Pfad steht, danach den Inhalt in einer eigenen Anfrage. Die WebSuite sucht in den Modellen des Projekts ein Objekt, dessen Namensspalte genau diesen Pfad enthält, und schreibt den Inhalt in dessen Inhaltsspalte. Findet sie keines, etwa weil das Modell kein Dateifeld mit `nameColumn` hat, lehnt sie die Datei ab, und QField verwirft sie.

### Einstellungen im QGIS-Formular

Das Formular richten Sie ein wie in den [Grundlagen](/admin-de/themen/fachmodule/qfieldcloud/grundlagen-qgis-projekte-fuer-qfield) beschrieben: In den Layereigenschaften unter *Attributformular* stellen Sie für die Namensspalte das Bedienelement *Anhang* (engl. *Attachment*) ein. Damit bietet QField an diesem Feld Kamera und Dateiauswahl an und trägt den Pfad der Datei ein.

Die Inhaltsspalte braucht im Formular kein Bedienelement. Blenden Sie sie aus, wenn sie über `useBlob` im Paket steht.

### Dateien zurück an die App

Ohne weitere Einstellung bleiben die Dateien auf dem Server: Die App kennt nur die Pfade, nicht die Inhalte. Sollen vorhandene Dateien im Feld sichtbar sein, nennen Sie das Dateifeld im Projekteintrag unter `syncFiles`:

```javascript
projects+ {
    ...
    syncFiles+ { field "photo_file" }
    syncFiles+ { field "scan_file" useBlob true }
}
```

Der Name muss genau ein Dateifeld in den Modellen des Projekts bezeichnen. Das Feld braucht `contentColumn` und, ohne `useBlob`, auch `nameColumn`. Andernfalls startet die Konfiguration nicht. Ins Paket kommt der Inhalt nur für Nutzer, die das Dateifeld lesen dürfen.

%warn
Mit `syncFiles` sieht jeder Nutzer, der das Projekt benutzen und das Dateifeld lesen darf, die Dateien aller Objekte im Paket. Dazu gehören auch die Fotos, die andere Nutzer aufgenommen haben.
%end

Es gibt zwei Arten der Übertragung:

- **Als Datei** (Vorgabe). Die WebSuite legt jede Datei unter dem Pfad aus der Namensspalte ins Paket, sodass das Bedienelement *Anhang* sie in QField anzeigt. Leere Dateien, absolute Pfade und Pfade mit `..` überspringt sie. Die Dateien hält sie im Cache des Projekts vor und liest sie erst wieder aus der Datenbank, wenn sich ihre Größe geändert hat.
- **Als Spalte im GeoPackage** (`useBlob true`). Der Inhalt steht in einer Binärspalte, die wie die `contentColumn` heißt. QField schickt bei jeder Änderung eines Objekts alle Attribute zurück, auch diese Spalte. Jede Bearbeitung überträgt die Datei also erneut. Direkt anzeigen kann QField eine Binärspalte nicht, ein Bild aber über ein HTML-Element im Formular als Data-URL:

```html
<script>
document.write(expression.evaluate("'<img src=\"data:image/jpeg;base64,' || to_base64(\"foto_content\") || '\">'"));
</script>
```

Neue Dateien nimmt auch in diesem Fall das Bedienelement *Anhang* an der Namensspalte auf. Sie werden wie oben beschrieben hochgeladen.

Ohne `useBlob` wird eine Datei in der Regel nur einmal übertragen und bleibt danach auf dem Gerät im Cache. Mit `useBlob` steckt sie im GeoPackage und wird bei jeder Synchronisierung erneut übertragen.

## Betrieb und Fehlersuche

Die WebSuite legt ihre Arbeitsdateien unter `<GWS_VAR_DIR>/qfieldcloud/projects/<uid>` ab. Dabei ist `<uid>` die Kennung des QField-Projekts, nicht die des WebSuite-Projekts. Dort liegen die fertigen Pakete als `package_<Zeitstempel>`, die Hintergrundkarte unter `cache` und die eingegangenen Änderungen in der Datei `deltas.sqlite`. Pakete, die älter als eine Stunde sind, löscht die WebSuite beim nächsten Packen. Änderungen löscht sie nach Ablauf von `deltaLifeTime` (Vorgabe 30 Tage) bei der nächsten Übertragung. Der Kartencache bleibt so lange gültig, wie `mapCacheLifeTime` vorgibt.

Ändern Sie das QGIS-Projekt, bemerkt die WebSuite das an dessen Inhalt und wertet es beim nächsten Zugriff neu aus. Ein Serverneustart ist dafür nicht nötig. Ein neues Paket entsteht allerdings erst, wenn die App eines anfordert.

Fehlt ein Layer in der App, hilft ein Blick ins Protokoll. Die WebSuite vermerkt dort, welchen Layer sie aus welchem Grund entfernt hat: nicht unterstützte Datenquelle, fehlender Tabellenname oder eine Tabelle, die in der Datenbank nicht gefunden wurde. Zum Prüfen ohne Gerät erzeugen Sie das Paket mit dem [Kommandozeilen-Befehl](/admin-de/konfiguration/cli/qfieldcloud) von Hand und sehen sich die Dateien an.

%info
**Besonderheit der WebSuite:** Bei Problemen braucht man beide Seiten. Was der Server abgelehnt hat und warum, steht im Protokoll der WebSuite. Was auf dem Gerät passiert ist, zeigen der Synchronisierungsstatus und das Nachrichtenprotokoll der App.
%end

%see
Siehe auch: [Aktion/qfieldcloud](/admin-de/konfiguration/action/qfieldcloud), [Modell/postgres](/admin-de/konfiguration/model/postgres), [Modell-Feld/file](/admin-de/konfiguration/modelField/file), [Kommandozeilen-Befehl/qfieldcloud](/admin-de/konfiguration/cli/qfieldcloud).
%end

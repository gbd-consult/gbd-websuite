# QField / QFieldCloud :/admin-de/themen/fachmodule/qfieldcloud

[QField](https://qfield.org) ist eine mobile, auf QGIS basierende Anwendung für die Erfassung von Geodaten im Außendienst. Die GBD WebSuite bildet die QFieldCloud-Schnittstelle selbst nach, sodass QField unmittelbar mit ihr synchronisiert und kein zusätzlicher QFieldCloud-Dienst nötig ist.

Für die App bleibt der Ablauf derselbe wie bei der echten QFieldCloud: anmelden, ein Datenpaket herunterladen, offline damit arbeiten, die Änderungen später zurückschicken. Anders ist nur, wer das Paket erzeugt. Die WebSuite baut es bei jeder Anforderung neu aus einem QGIS-Projekt und den Datenbanktabellen, die dahinter liegen, und schreibt die zurückgemeldeten Änderungen über ihre Datenmodelle wieder in dieselben Tabellen.

Die Einrichtung besteht aus zwei Teilen, die zusammenpassen müssen: Im QGIS-Projekt legen Sie mit der Erweiterung QFieldSync fest, was mit welchem Layer geschehen soll; in der WebSuite konfigurieren Sie eine Aktion, die dieses Projekt anbindet und die Berechtigungen setzt.

## Voraussetzungen

- Ein QGIS-Projekt, entweder als Datei oder in einer Datenbank abgelegt. Sollen Anhänge oder weitere Dateien mit ins Paket, muss es eine Datei sein – nur dann lassen sich die in QFieldSync relativ angegebenen Verzeichnisse auflösen.
- Alle Layer, die im Feld bearbeitet werden sollen, müssen aus PostgreSQL/PostGIS stammen. Andere Quellen lassen sich nicht offline editieren; die WebSuite entfernt solche Layer aus dem Paket und schreibt eine Warnung ins Protokoll.
- Die QGIS-Erweiterung QFieldSync, mit der Sie das Projekt vorbereiten.
- Benutzerkonten für die Feldkräfte. Die App meldet sich mit denselben Zugangsdaten an wie ein Browser-Nutzer, geprüft über die konfigurierten Provider der [](/admin-de/themen/zugriff/authentifizierung). Dazu eine dauerhafte Sitzungsverwaltung.

## Das QGIS-Projekt mit QFieldSync vorbereiten

QFieldSync hinterlegt seine Einstellungen als Eigenschaften im Projekt. Die WebSuite liest genau diese Eigenschaften aus und richtet sich danach – Sie konfigurieren das Verhalten der Layer also in QGIS, nicht in der WebSuite. Ausgewertet werden dabei die Angaben, die das Paket betreffen: die Aktion je Layer, die Hintergrundkarte, das Bearbeitungsgebiet und die mitzukopierenden Verzeichnisse. Alle übrigen QFieldSync-Einstellungen – etwa zu Geofencing, Positionsaufzeichnung oder Bildstempeln – bleiben unangetastet im Projekt stehen und wertet QField selbst auf dem Gerät aus.

Je Layer entscheidet die eingestellte Aktion über sein Schicksal im Paket:

- **Offline bearbeiten** – der Layer wird editierbar. Die WebSuite exportiert seine Objekte in ein GeoPackage, das mit ins Paket kommt, und lenkt den Layer im mitgelieferten Projekt auf diese Datei um. Das setzt eine PostGIS-Tabelle voraus, und zwar eine echte Tabelle: Layer, die auf einer Unterabfrage oder einem `SELECT` beruhen, lassen sich nicht zuordnen und werden entfernt.
- **Entfernen** – der Layer wird aus dem mitgelieferten Projekt gelöscht. Gruppen, die dadurch leer werden, verschwinden ebenfalls.
- **Keine Aktion** – der Layer bleibt unverändert stehen, mitsamt seiner ursprünglichen Datenquelle. Im Feld ist er damit nur nutzbar, solange das Gerät diese Quelle erreicht. Für den Offline-Einsatz ist das die häufigste Fehlerquelle.

Aktivieren Sie in den Layereigenschaften unter *QField* > *Befugnisse* alle vier Optionen (*Erstellen von Objekt deaktivieren*, *Löschen von Objekten deaktivieren*, *Bearbeiten von Attributen deaktivieren*, *Bearbeiten von Geometrien deaktivieren*), behandelt die WebSuite den Layer als reine Lesequelle. Er wird dann zwar ins Paket exportiert, muss aber kein editierbares Datenmodell besitzen. Einzelne Befugnisse verrechnet die WebSuite mit den Rechten am Datenmodell, wie unter [](/admin-de/themen/fachmodule/qfieldcloud/berechtigungen) beschrieben.

### Hintergrundkarte

Ist in QFieldSync die Erzeugung einer Hintergrundkarte eingeschaltet, rendert der QGIS-Server sie und die WebSuite legt sie als einzelnes Rasterbild ins Paket. Als Vorlage dient wahlweise ein Kartenthema des Projekts oder ein einzelner Layer. Aus den eingestellten Zoomstufen nimmt die WebSuite die höhere und begrenzt sie auf den Bereich 3 bis 20; daraus ergibt sich die Auflösung des Bildes.

Das Rendern ist der mit Abstand aufwendigste Teil der Paketerstellung. Halten Sie das Gebiet klein und die Zoomstufe so niedrig wie vertretbar, und schalten Sie in der Aktion einen Cache ein.

### Bearbeitungsgebiet

Legen Sie in QFieldSync ein Bearbeitungsgebiet fest, dient es der WebSuite als Ausschnitt für die Hintergrundkarte. Ist zusätzlich eingestellt, dass nur dieses Gebiet kopiert werden soll, exportiert sie auch die Objekte der editierbaren Layer nur aus diesem Bereich. Ohne Bearbeitungsgebiet gilt die Ausdehnung des QGIS-Projekts – bei einem landesweiten Datenbestand entsprechend viel.

### Anhänge und mitgelieferte Dateien

Die in QFieldSync angegebenen Verzeichnisse für Anhänge und Daten kopiert die WebSuite unverändert ins Paket. Relative Pfade bezieht sie auf den Ort der Projektdatei. Verschachtelte Angaben werden zusammengefasst, sodass kein Verzeichnis doppelt im Paket landet.

## Die Aktion einrichten

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

Der Abschnitt `models` ist dabei **nicht** zwingend. Findet die WebSuite zu einer editierbaren Tabelle kein konfiguriertes Modell, legt sie selbst eines an, das alle Spalten übernimmt und Bearbeiten zulässt. Für einen ersten Versuch genügt daher `provider.path`. Ein eigenes Modell brauchen Sie, sobald Sie die Felder einschränken, die Bearbeitung auf bestimmte Rollen begrenzen oder Dateianhänge in der Datenbank ablegen wollen.

Alle Optionen samt Beispielen führt die [Konfigurationsseite der Aktion](/admin-de/konfiguration/action/qfieldcloud) auf.

## Zugang für die App

Die Aktion beantwortet die Anfragen der App unter einem internen Befehl. Damit die Nutzer eine gewöhnliche Adresse eintragen können, legen Sie in der Webseite eine Rewrite-Regel an, die einen Pfad Ihrer Wahl darauf abbildet. In der App wählen die Nutzer dann *QFieldCloud-Projekte*, tragen diese Adresse als Server ein und melden sich mit ihren WebSuite-Zugangsdaten an. Anschließend laden sie das Projekt aus der Liste herunter und übertragen ihre Änderungen später über *Synchronisieren* zurück.

Die Anmeldung selbst müssen Sie nicht einrichten. Die Aktion bringt eine eigene Authentifizierungsmethode mit und registriert sie beim Start; sie taucht deshalb nicht unter den Methoden auf, die Sie in der Applikation auswählen. Diese Methode prüft die von der App übermittelten Zugangsdaten über die Authentifizierungs-Provider und gibt ein Token zurück, das die Kennung einer Sitzung ist. Jede weitere Anfrage der App weist dieses Token vor und verlängert damit die Sitzung; meldet sich ein Nutzer in der App ab, wird sie gelöscht.

Die Lebensdauer eines Tokens ist also die einer Sitzung. Richten Sie deshalb eine dauerhafte [](/admin-de/konfiguration/authSessionManager) ein – ohne sie gehen alle Anmeldungen bei jedem Serverneustart verloren, und die Feldkräfte stehen im ungünstigsten Moment ohne Zugang da.

## Was im Datenpaket steckt

Fordert die App ein Paket an, legt die WebSuite einen Hintergrundauftrag an und baut es zusammen:

- je editierbarem Layer ein GeoPackage mit den exportierten Objekten,
- die gerenderte Hintergrundkarte als Rasterbild,
- die kopierten Anhang- und Datenverzeichnisse,
- das QGIS-Projekt selbst, umgeschrieben auf die enthaltenen Dateien.

Beim Umschreiben des Projekts ersetzt die WebSuite die Datenquellen der editierbaren Layer und der Hintergrundkarte durch die lokalen Dateien, entfernt die ausgeschlossenen Layer und stellt das Projekt auf relative Pfade um. Auch Beziehungen zwischen Layern und Widgets vom Typ *Relation Reference* werden mitgezogen, sodass verknüpfte Objekte im Feld weiter funktionieren.

Zwei Eigenheiten des Exports sollten Sie kennen. Erstens landen nur Attribute der Typen Wahrheitswert, Datum, Zeit, Zeitstempel, Ganzzahl, Fließkommazahl und Text im GeoPackage; Felder anderer Typen bleiben außen vor; wie Dateifelder behandelt werden, beschreibt der Abschnitt [](/admin-de/themen/fachmodule/qfieldcloud/arbeiten-mit-dateien). Zweitens benennt die WebSuite ein Feld namens `fid` im Paket in `fid_gws` um, weil GDAL diesen Namen selbst belegt; auf dem Rückweg macht sie das wieder rückgängig.

Enthält das QGIS-Projekt die Platzhalter `{user.authToken}`, `{user.loginName}` oder `{user.displayName}`, setzt die WebSuite beim Packen die Werte des anfordernden Nutzers ein. So kann ein Layer im Feld etwa einen Dienst der WebSuite selbst ansprechen, ohne dass Zugangsdaten im Projekt stehen.

## Der Rückweg: Änderungen und Anhänge

QField schickt die im Feld vorgenommenen Änderungen als Liste von Einzeloperationen zurück – anlegen, ändern, löschen. Die WebSuite ordnet jede Operation dem Datenmodell des betroffenen Layers zu und führt sie darüber aus. Damit gelten dieselben Regeln wie beim [](/admin-de/themen/daten/editieren) im Browser: Validatoren, Wertvorgaben und Berechtigungen greifen unverändert. Alle Operationen einer Übertragung laufen gemeinsam in einer Transaktion: Schlägt eine fehl, wird nichts geschrieben.

Dateianhänge kommen getrennt hinterher, wie unter [](/admin-de/themen/fachmodule/qfieldcloud/arbeiten-mit-dateien) beschrieben.

## Berechtigungen

Es greifen drei Ebenen, und alle drei sollten Sie bewusst setzen:

- Die **Aktion** selbst legt fest, wer die Schnittstelle überhaupt ansprechen darf.
- Jeder **Projekteintrag** hat eigene Zugriffsregeln. In der App sieht ein Nutzer nur die Projekte, die er benutzen darf.
- Die **Layer** bestimmen, was bearbeitet werden darf: über die *Befugnisse* in QFieldSync, über die Rechte an den Modellen oder über beides.

Für die Bearbeitung gibt es damit zwei Wege. Sollen alle Nutzer eines Projekts dieselben Möglichkeiten haben, genügen die Befugnisse in QFieldSync. Brauchen verschiedene Nutzergruppen verschiedene Rechte, steuern Sie die Bearbeitung über die Modelle der WebSuite.

### Berechtigungen mit QFieldSync steuern

Sie konfigurieren keine eigenen Modelle, oder Modelle, die allen Nutzern die Bearbeitung erlauben (`permissions.edit "allow user"`). Die automatisch erzeugten Modelle sind für jeden bearbeitbar, der das Projekt benutzen darf; wer zugreift, entscheiden dann allein die Zugriffsregeln der Aktion und des Projekteintrags.

Was bearbeitet werden darf, legen Sie je Layer in den Layereigenschaften unter *QField* > *Befugnisse* fest:

- *Erstellen von Objekt deaktivieren*
- *Löschen von Objekten deaktivieren*
- *Bearbeiten von Attributen deaktivieren*
- *Bearbeiten von Geometrien deaktivieren*

Diese Einstellungen gelten für alle Nutzer gleich. QField bietet die deaktivierten Werkzeuge nicht an, und die WebSuite prüft sie beim Zurückschicken erneut: Eine Änderung, deren Befugnis im QGIS-Projekt deaktiviert ist, weist sie ab. Da alle Operationen einer Übertragung in einer Transaktion laufen, wird dann die ganze Übertragung abgelehnt.

Befugnisse, die über einen Ausdruck gesteuert werden (das Symbol neben der Option), wertet die WebSuite nicht aus. Der Ausdruck bleibt im Projekt stehen und wird nur von QField auf dem Gerät ausgewertet; beim Zurückschicken prüft die WebSuite ihn nicht.

### Berechtigungen mit der WebSuite steuern

Sie konfigurieren Modelle mit `isEditable` und vergeben die Rechte getrennt nach Rollen (`permissions.create`, `permissions.write`, `permissions.delete`). So kann etwa eine Gruppe Objekte nur ändern, eine andere auch anlegen und löschen.

Die Befugnisse in QFieldSync gelten dabei weiter. Eine Bearbeitung ist nur erlaubt, wenn das Modell sie dem Nutzer zulässt und die zugehörige Befugnis nicht deaktiviert ist:

| Befugnis in QFieldSync | nötiges Recht am Modell |
|---|---|
| *Erstellen von Objekt deaktivieren* | `permissions.create` |
| *Löschen von Objekten deaktivieren* | `permissions.delete` |
| *Bearbeiten von Attributen deaktivieren* | `permissions.write` |
| *Bearbeiten von Geometrien deaktivieren* | `permissions.write` am Modell und am Geometriefeld |

Ohne `isEditable` am Modell, oder wenn der Layer in QGIS schreibgeschützt ist, ist jede Bearbeitung deaktiviert. Das Ergebnis berechnet die WebSuite beim Packen für den anfordernden Nutzer und setzt die Befugnisse im mitgelieferten Projekt entsprechend. Ist jede Bearbeitung deaktiviert, kommt der Layer schreibgeschützt ins Paket. Zwei Feldkräfte können daher vom selben QGIS-Projekt unterschiedliche Pakete erhalten.

Beim Zurückschicken greifen die Rechte am Modell wie beim Editieren im Browser. Was serverseitig verbindlich ausgeschlossen sein soll, regeln Sie hier und nicht über einen Ausdruck in QFieldSync.

## Arbeiten mit Dateien

Fotos und andere Anhänge, die im Feld entstehen, legt die WebSuite in der Datenbank ab, in derselben Tabelle wie das Objekt. Auf Wunsch schickt sie vorhandene Dateien auch mit dem Paket an die App.

### Spalten in der Datenbank

Je Anhang braucht die Tabelle zwei Spalten:

- eine Textspalte für den Dateinamen, genauer den Pfad, den QField für die Datei vergibt, etwa `DCIM/foto_20261016.jpg`,
- eine Spalte vom Typ `bytea` für den Dateiinhalt.

Die Namensspalte ist ein gewöhnliches Attribut: Sie wandert ins GeoPackage, QField trägt dort den Pfad ein und schickt ihn mit den übrigen Änderungen zurück. Die Inhaltsspalte kommt nicht ins Paket, solange Sie sie nicht über `syncFiles` mit `useBlob` anfordern (siehe unten).

### Das Dateifeld im Modell

Damit die WebSuite die beiden Spalten als Anhang versteht, braucht das Modell ein Feld vom Typ [`file`](/admin-de/konfiguration/modelField/file), das sie verbindet. Das Feld ist virtuell: Sein Name entspricht keiner Spalte, `nameColumn` und `contentColumn` verweisen auf die beiden echten Spalten. Die Namensspalte bleibt daneben als eigenes Textfeld im Modell, etwa über `withAutoFields`:

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

Im QGIS-Projekt stellen Sie in den Layereigenschaften unter *Attributformular* für die Namensspalte das Bedienelement *Anhang* (engl. *Attachment*) ein. Damit bietet QField an diesem Feld Kamera und Dateiauswahl an und trägt den Pfad der Datei ein. Speichern Sie den Pfad relativ zum Projekt: Mit diesem Pfad ordnet die WebSuite die Datei zu, und nur relative Pfade schickt sie im Paket mit zurück auf das Gerät.

Die Inhaltsspalte braucht im Formular kein Bedienelement. Blenden Sie sie aus, sofern sie über `useBlob` im Paket steht.

### Dateien zurück an die App

Ohne weitere Einstellung bleiben die Dateien auf dem Server: Die App kennt nur die Pfade, nicht die Inhalte. Sollen vorhandene Dateien im Feld sichtbar sein, nennen Sie das Dateifeld im Projekteintrag unter `syncFiles`:

```javascript
projects+ {
    ...
    syncFiles+ { field "photo_file" }
    syncFiles+ { field "scan_file" useBlob true }
}
```

Der Name muss genau ein Dateifeld in den Modellen des Projekts bezeichnen; das Feld braucht `contentColumn` und, ohne `useBlob`, auch `nameColumn`. Andernfalls startet die Konfiguration nicht. Ins Paket kommt der Inhalt nur für Nutzer, die das Dateifeld lesen dürfen.

%warn
Mit `syncFiles` sieht jeder Nutzer, der das Projekt benutzen und das Dateifeld lesen darf, die Dateien aller Objekte im Paket, also auch die Fotos, die andere Nutzer aufgenommen haben.
%end

Es gibt zwei Arten der Übertragung:

- **Als Datei** (Vorgabe). Die WebSuite legt jede Datei unter dem Pfad aus der Namensspalte ins Paket, sodass das Bedienelement *Anhang* sie in QField anzeigt. Leere Dateien, absolute Pfade und Pfade mit `..` überspringt sie. Die Dateien hält sie im Cache des Projekts vor und liest sie erst wieder aus der Datenbank, wenn sich ihre Größe geändert hat.
- **Als Spalte im GeoPackage** (`useBlob true`). Der Inhalt steht in einer Binärspalte, die wie die `contentColumn` heißt. QField schickt bei jeder Änderung eines Objekts alle Attribute zurück, auch diese Spalte; jede Bearbeitung überträgt die Datei also erneut. Anzeigen kann QField eine Binärspalte nicht direkt, ein Bild aber über ein HTML-Element im Formular als Data-URL:

```html
<img src=expression.evaluate("'data:image/jpeg;base64,' || to_base64(scan_content)")>
```

Neue Dateien nimmt auch in diesem Fall das Bedienelement *Anhang* an der Namensspalte auf, und sie werden wie oben beschrieben hochgeladen.

Ohne `useBlob` wird eine Datei in der Regel nur einmal übertragen und bleibt danach auf dem Gerät im Cache. Mit `useBlob` steckt sie im GeoPackage und wird bei jeder Synchronisierung erneut übertragen.

## Betrieb und Fehlersuche

Die WebSuite legt ihre Arbeitsdateien unter `<GWS_VAR_DIR>/qfieldcloud/projects/<uid>` ab, wobei `<uid>` die Kennung des QField-Projekts ist und nicht die des WebSuite-Projekts: die fertigen Pakete als `package_<Zeitstempel>`, die Hintergrundkarte unter `cache`, die eingegangenen Änderungen in der Datei `deltas.sqlite`. Pakete, die älter als eine Stunde sind, räumt die WebSuite beim nächsten Packen weg, Änderungen nach Ablauf von `deltaLifeTime` (Vorgabe 30 Tage) bei der nächsten Übertragung; der Kartencache bleibt so lange gültig, wie es `mapCacheLifeTime` vorgibt.

Ändern Sie das QGIS-Projekt, bemerkt die WebSuite das an dessen Inhalt und wertet es beim nächsten Zugriff neu aus – ein Serverneustart ist dafür nicht nötig. Ein neues Paket entsteht allerdings erst, wenn die App eines anfordert.

Bleibt ein Layer in der App unerwartet aus, hilft ein Blick ins Protokoll: Die WebSuite vermerkt dort, welchen Layer sie aus welchem Grund entfernt hat – nicht unterstützte Datenquelle, fehlender Tabellenname oder eine Tabelle, die in der Datenbank nicht gefunden wurde. Zum Prüfen, ohne ein Gerät zu bemühen, erzeugen Sie das Paket mit dem [Kommandozeilen-Befehl](/admin-de/konfiguration/cli/qfieldcloud) von Hand und sehen sich die Dateien an.

%see
Siehe auch: [Aktion/qfieldcloud](/admin-de/konfiguration/action/qfieldcloud), [Modell/postgres](/admin-de/konfiguration/model/postgres), [Modell-Feld/file](/admin-de/konfiguration/modelField/file), [Kommandozeilen-Befehl/qfieldcloud](/admin-de/konfiguration/cli/qfieldcloud).
%end

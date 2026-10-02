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

Sperren Sie in QFieldSync alle vier Bearbeitungsmöglichkeiten eines Layers – Attribute, Geometrie, Anlegen und Löschen –, behandelt die WebSuite ihn als reine Lesequelle. Er wird dann zwar ins Paket exportiert, muss aber kein editierbares Datenmodell besitzen.

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

Zwei Eigenheiten des Exports sollten Sie kennen. Erstens landen nur Attribute der Typen Wahrheitswert, Datum, Zeit, Zeitstempel, Ganzzahl, Fließkommazahl und Text im GeoPackage; Felder anderer Typen – etwa Dateifelder – bleiben außen vor und werden getrennt behandelt. Zweitens benennt die WebSuite ein Feld namens `fid` im Paket in `fid_gws` um, weil GDAL diesen Namen selbst belegt; auf dem Rückweg macht sie das wieder rückgängig.

Enthält das QGIS-Projekt die Platzhalter `{user.authToken}`, `{user.loginName}` oder `{user.displayName}`, setzt die WebSuite beim Packen die Werte des anfordernden Nutzers ein. So kann ein Layer im Feld etwa einen Dienst der WebSuite selbst ansprechen, ohne dass Zugangsdaten im Projekt stehen.

## Der Rückweg: Änderungen und Anhänge

QField schickt die im Feld vorgenommenen Änderungen als Liste von Einzeloperationen zurück – anlegen, ändern, löschen. Die WebSuite ordnet jede Operation dem Datenmodell des betroffenen Layers zu und führt sie darüber aus. Damit gelten dieselben Regeln wie beim [](/admin-de/themen/daten/editieren) im Browser: Validatoren, Wertvorgaben und Berechtigungen greifen unverändert. Die Operationen eines Modells laufen gemeinsam in einer Transaktion.

Dateianhänge kommen getrennt hinterher. QField überträgt zuerst die Änderung am Objekt, in der nur der Dateiname steht, danach den Inhalt in einer eigenen Anfrage. Die WebSuite findet das zugehörige Objekt, indem sie diesen Namen mit dem Namensfeld des Dateifelds vergleicht. Ist an dem Modell kein Dateifeld mit Namensspalte konfiguriert, lässt sich der Anhang nicht zuordnen und geht verloren.

## Berechtigungen

Es greifen drei Ebenen, und alle drei sollten Sie bewusst setzen:

- Die **Aktion** selbst legt fest, wer die Schnittstelle überhaupt ansprechen darf.
- Jeder **Projekteintrag** hat eigene Zugriffsregeln. In der App sieht ein Nutzer nur die Projekte, die er benutzen darf.
- Die **Modelle** entscheiden über das Schreiben. `isEditable` erlaubt die Bearbeitung grundsätzlich, `permissions.edit` schränkt sie auf Rollen ein.

Verzichten Sie auf eigene Modelle, sind die automatisch erzeugten uneingeschränkt bearbeitbar. Die Zugriffsregeln des Projekteintrags sind dann die einzige Schranke – für einen Produktivbetrieb zu wenig.

## Betrieb und Fehlersuche

Die WebSuite legt ihre Arbeitsdateien unter `<GWS_VAR_DIR>/qfieldcloud/projects/<uid>` ab, wobei `<uid>` die Kennung des QField-Projekts ist und nicht die des WebSuite-Projekts: die fertigen Pakete als `package_<Zeitstempel>`, die Kacheln der Hintergrundkarte unter `cache`, die eingegangenen Änderungslisten unter `deltas`. Pakete und Änderungslisten, die älter als eine Stunde sind, räumt die WebSuite beim nächsten Packen weg; der Kartencache bleibt so lange gültig, wie es `mapCacheLifeTime` vorgibt.

Ändern Sie das QGIS-Projekt, bemerkt die WebSuite das an dessen Inhalt und wertet es beim nächsten Zugriff neu aus – ein Serverneustart ist dafür nicht nötig. Ein neues Paket entsteht allerdings erst, wenn die App eines anfordert.

Bleibt ein Layer in der App unerwartet aus, hilft ein Blick ins Protokoll: Die WebSuite vermerkt dort, welchen Layer sie aus welchem Grund entfernt hat – nicht unterstützte Datenquelle, fehlender Tabellenname, kein Modell oder ein nicht editierbares Modell. Zum Prüfen, ohne ein Gerät zu bemühen, erzeugen Sie das Paket mit dem [Kommandozeilen-Befehl](/admin-de/konfiguration/cli/qfieldcloud) von Hand und sehen sich die Dateien an.

%see
Siehe auch: [Aktion/qfieldcloud](/admin-de/konfiguration/action/qfieldcloud), [Modell/postgres](/admin-de/konfiguration/model/postgres), [Modell-Feld/file](/admin-de/konfiguration/modelField/file), [Kommandozeilen-Befehl/qfieldcloud](/admin-de/konfiguration/cli/qfieldcloud).
%end

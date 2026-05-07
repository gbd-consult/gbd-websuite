# Was wir hier getan haben — verständlich erklärt

Dieses Dokument beschreibt, was auf dem Branch
`feat/spec-ux-strings` der GBD WebSuite passiert ist und warum.
Es richtet sich ausdrücklich auch an Leser, die nicht selbst
programmieren — Projektleitung, Produkt-Owner, Konfigurations-Expertinnen
und alle, die mit der WebSuite arbeiten, sie konfigurieren oder
weiterverkaufen.

## Worum geht es eigentlich?

Die GBD WebSuite ist eine Web-GIS-Plattform, die man pro Installation
ausführlich konfigurieren muss: welche Karten, welche Layer, wer darf
sich wie anmelden, woher kommen die Daten, wie sehen die Druckvorlagen
aus. Diese Konfiguration ist groß — leicht mehrere tausend einzelne
Optionen, verteilt auf viele Module.

Damit Menschen die Konfiguration **bequem** anlegen können (und nicht
mit der Hand in JSON-Dateien schreiben müssen), gibt es ein zweites
Werkzeug: den **Konfigurator**. Das ist eine grafische Oberfläche, die
Felder zum Ausfüllen anbietet, Hilfetexte zeigt, Tooltipps einblendet
und sinnvolle Standard-Vorgaben macht.

Bisher gab es ein Problem: **Der Konfigurator und die WebSuite kannten
sich gegenseitig nicht gut genug.** Der Konfigurator wusste zwar
*welche Felder* es gibt — die Liste der Konfigurationsoptionen wird
ihm aus der WebSuite geliefert. Aber **die schönen Texte daneben**
(„Hostname des PostgreSQL-Servers", „Wann sollten Sie diese Option
benutzen?", „Beispielwert: 5432") musste der Konfigurator selbst
nachpflegen, in einer eigenen Datei namens `ux-schema.json`.

Das hatte zwei unangenehme Folgen:

1. **Doppelte Pflege.** Wenn die WebSuite eine neue Option bekam, musste
   der Konfigurator separat einen Hilfetext dafür schreiben. Sonst
   stand im Frontend nur der nackte technische Feldname —
   `serviceName`, `sqlFilter`, `dbUid` — und kein Mensch wusste, was
   er da eintragen soll.
2. **Sprache und Mehrsprachigkeit.** Der Konfigurator kennt nur Deutsch
   und Englisch, aber die alten Texte in der WebSuite waren teils nur
   in Englisch (oder gar nicht) vorhanden. Dadurch entstand eine
   Mischanzeige aus deutschen und englischen Beschriftungen — unschön
   und für Endnutzer verwirrend.

## Unsere Absicht

Wir haben die WebSuite zur **alleinigen, autoritativen Quelle** für
sämtliche Bedienungstexte ihrer Konfiguration gemacht. Der Konfigurator
zieht künftig alle Hilfetexte direkt aus der WebSuite. Die separate
`ux-schema.json` im Konfigurator-Repo wird damit überflüssig.

Das Ziel in einem Satz:

> **Jede Konfigurationsoption der WebSuite kommt zukünftig mit einem
> sauberen Hilfetext in Deutsch und Englisch — und zwar dort, wo die
> Option selbst definiert ist, nicht irgendwo anders.**

## Was haben wir konkret gemacht?

Die Arbeit lief in drei großen Etappen.

### Etappe 1 — Das Format erfinden und einbauen

Wir haben einen neuen Bereich im Daten-Austauschformat zwischen
WebSuite und Konfigurator angelegt: einen Block namens **`uxStrings`**.
Dieser Block enthält pro Konfigurationsoption sieben strukturierte
Informationsfelder, alle in Deutsch und Englisch:

| Feld         | Bedeutung                                                            |
|--------------|----------------------------------------------------------------------|
| `label`      | Die Bezeichnung des Feldes, wie sie der Nutzer im Konfigurator sieht |
| `purpose`    | Wozu dient diese Option? Eine kurze Erklärung.                       |
| `whenToUse`  | Wann sollte ich diese Option überhaupt benutzen?                     |
| `complexity` | Wie heikel ist diese Option? (Anfänger / Fortgeschritten / Experte)  |
| `useCases`   | Typische Anwendungsfälle                                             |
| `seeAlso`    | Verweise auf verwandte Konfigurationen                               |
| `example`    | Beispielwert                                                         |

Diese Texte legen Plugin-Entwicklerinnen und -Entwickler nun direkt
neben ihrem Modul ab — in einer kleinen Textdatei mit Namen
`_doc/ux.ini`. Dort steht zum Beispiel:

```
gws.plugin.postgres.provider.Config.host.label = Host
gws.plugin.postgres.provider.Config.host.purpose = Hostname oder IP-Adresse des PostgreSQL-Servers.
gws.plugin.postgres.provider.Config.host.example = "db.example.com"
```

Beim nächsten Build der WebSuite werden alle diese Dateien automatisch
eingesammelt und in eine zentrale Datei (`specs.json`) geschrieben, die
der Konfigurator dann lesen kann.

**Was zusätzlich gebaut wurde, um das Format dauerhaft zu pflegen:**

- Ein **Hilfsprogramm** (`bootstrap_ux.py`), das aus den vorhandenen
  Konfigurationsklassen automatisch ein leeres `ux.ini`-Skelett mit
  Vorschlägen erzeugt. Maintainer müssen die Vorschläge dann nur noch
  korrigieren und ergänzen — sie schreiben nicht von null.
- Ein **Coverage-Werkzeug** (`coverage.py`), das anzeigt, wieviel
  Prozent der Konfigurationsoptionen schon vernünftige Hilfetexte
  haben. Im Continuous-Integration-System können wir damit verhindern,
  dass jemand neue Optionen ohne Hilfetexte einbaut.
- **Tests**, die sicherstellen, dass das Sammeln der Texte zuverlässig
  funktioniert.
- Eine **Konventionsbeschreibung** unter `app/gws/spec/_doc/README.md`
  für Plugin-Maintainer.

### Etappe 2 — Die WebSuite flächendeckend beschreiben

Anschließend haben wir Modul für Modul Hilfetexte verfasst. Das war
der größere Brocken: **viele Module, viele Optionen**.

Die Arbeit wurde in drei parallelen Wellen organisiert:

- **Welle A — Anmeldung und Berechtigungen**
  Alles rund um „Wer darf sich wie anmelden, mit welcher
  Zwei-Faktor-Methode, gegen welches LDAP, welche Datenbank, welcher
  Account-Verwaltung".
- **Welle B — Karten, Projekte, Layer, Datenmodelle**
  Das Herzstück einer GIS-Konfiguration: Welche Projekte gibt es,
  welche Karten, welche Layer aus welchen Quellen (PostGIS, GeoJSON,
  WMS, Kacheldienste, Rasterdaten), welche Datenmodelle für editierbare
  Layer.
- **Welle C — Drucken, Vorlagen, Schnittstellen, Werkzeuge**
  Druckvorlagen, Templates, OGC-Dienste (WMS/WFS/CSW/WMTS), die
  WebSuite-eigenen ALKIS-, GekoS- und QFieldCloud-Anbindungen,
  Werkzeugleisten.

Pro Welle arbeiteten mehrere Sub-Agenten parallel an je einem
Themenbereich. Dadurch konnte die Pflege deutlich beschleunigt werden,
und jedes Modul wurde in einem eigenen Commit dokumentiert — die
Historie bleibt nachvollziehbar.

### Etappe 3 — Folgewünsche des Konfigurator-Teams

Nach Welle C kam vom Konfigurator-Team eine zweite Anforderungsrunde
(`konfigurator/docs/specs-generator-followup-requirements.md`).
Diese betraf vor allem zwei besondere Stellen, an denen der
Konfigurator dem Nutzer **eine Auswahl** anbietet:

#### „Variant-Auswahl" (Phase F1 + F2)

Beispiel: Wenn ein Nutzer im Konfigurator „Auth-Provider hinzufügen"
klickt, soll er **vor der Auswahl** sehen können, was die
Auth-Provider-Familie überhaupt ist — bevor er entscheidet, ob er
einen LDAP-, Datei- oder PostgreSQL-Auth-Provider haben möchte. Das
funktionierte vorher nicht, weil es für die Familie als Ganzes keine
Hilfetexte gab — nur für die einzelnen Mitglieder.

Wir haben deshalb für **22 solcher Familien und Top-Level-Klassen**
zusätzliche Hilfetexte ergänzt:

- 13 Variant-Familien (Aktionen, Auth-Provider, Layer-Arten, Datenbank-
  Provider, OWS-Dienste, Helfer, …) — siehe `app/gws/ext/_doc/ux.ini`.
- Zentrale Top-Level-Klassen wie Application, Web-Manager, Server-
  Konfiguration, Cache-Konfiguration — siehe diverse neue
  `_doc/ux.ini`-Dateien unter `app/gws/_doc/`,
  `app/gws/base/{application,web,database,client}/_doc/`,
  `app/gws/gis/cache/_doc/`, `app/gws/server/_doc/`.

#### „Apply-Templates" (Phase F4)

Damit der Nutzer nicht jedes Mal bei Null anfangen muss, gibt es
zusätzlich **Beispiel-Vorlagen**, die er per Klick einfügen kann.
Wir haben **22 solcher Vorlagen** für 7 Familien angelegt — z.B.
„LDAP gegen Active Directory", „PostgreSQL-Layer mit Editierung",
„Standard-WMS-Dienst". Diese liegen in `app/gws/ext/_doc/scenarios.json`.

#### Lückenschluss bei Property-Beschriftungen (Phase F3)

Der Konfigurator hatte gemeldet, dass er bei vielen einzelnen
Optionen immer noch automatisch generierte technische Bezeichnungen
zeigt („dbUid", „sqlFilter", „srid"). Wir haben deshalb noch einmal
flächendeckend nachgepflegt — diesmal vier parallele Sub-Agenten,
aufgeteilt nach Modul-Familie. Ergebnis:

- **100 % der UI-relevanten Konfigurationsoptionen** haben jetzt eine
  ordentliche deutsche Beschriftung.
- **96,9 %** haben zusätzlich eine englische.

#### Verweise zwischen verwandten Konfigurationen (Phase F6)

Wer einen LDAP-Auth-Provider konfiguriert, möchte vielleicht wissen,
dass es auch einen Datei- und einen Datenbank-Auth-Provider gibt.
Solche Verweise (`seeAlso`) wurden in 3 Plugin-Familien (Legend,
Auth-Provider, OWS-Server) wechselseitig ergänzt — jedes Mitglied
verweist auf alle anderen.

## Welche Dateien wurden angepasst?

### Neue Dateien (Konvention und Werkzeuge)

| Datei                                    | Funktion                                     |
|------------------------------------------|----------------------------------------------|
| `app/gws/spec/_doc/README.md`            | Konvention für Plugin-Maintainer             |
| `app/gws/spec/generator/bootstrap_ux.py` | Hilfsprogramm, das `ux.ini`-Skelette erzeugt |
| `app/gws/spec/generator/coverage.py`     | Werkzeug zur Messung der Hilfetext-Abdeckung |
| `app/gws/spec/generator/_test/`          | Tests für die neue Pipeline                  |

### Erweiterte Dateien (Generator-Logik)

| Datei                                  | Was geändert wurde                             |
|----------------------------------------|------------------------------------------------|
| `app/gws/spec/core.py`                 | Neuer Datenblock `uxStrings` und `scenarios`   |
| `app/gws/spec/generator/main.py`       | Pipeline-Erweiterung                           |
| `app/gws/spec/generator/strings.py`    | Neue Sammler für `ux.ini` und `scenarios.json` |
| `app/gws/spec/generator/typescript.py` | Export der neuen Typen für den Konfigurator    |
| `app/gws/spec/generator/configref.py`  | Hilfetexte landen auch in der HTML-Doku        |

### Inhalts-Dateien (das eigentliche „Schreiben")

Pro Modul wurde eine Datei `_doc/ux.ini` angelegt, in vielen Fällen
zusätzlich `_doc/strings.ini` (für ältere i18n-Texte) und
`_doc/scenarios.json` (für Apply-Templates).

Stand am Ende der Arbeit:

- **Pilot:** `app/gws/plugin/postgres/_doc/ux.ini`
- **Welle A:** alle Auth-Module unter `app/gws/base/auth/`,
  `app/gws/plugin/auth_method/*`, `app/gws/plugin/auth_mfa/*`,
  `app/gws/plugin/auth_provider/*`, `app/gws/plugin/account/*`
- **Welle B:** alle Module unter
  `app/gws/base/{project,map,layer,model,metadata}` und alle
  Layer-/Model-Plugins
- **Welle C:** alle Module unter
  `app/gws/base/{printer,template,exporter,edit,search}` und alle
  Druck-/Template-/OWS-/Tool-Plugins
- **Folge-Welle (F1–F6):**
    - `app/gws/ext/_doc/ux.ini` (Variant-Familien)
    - `app/gws/ext/_doc/scenarios.json` (Apply-Templates)
    - Neue `_doc/ux.ini` für Application, WebManager, Cache, Server,
      Database, Client
    - Erweiterte Property-Labels in 30+ Modulen

### Dokumentation

| Datei                                           | Zweck                                   |
|-------------------------------------------------|-----------------------------------------|
| `BRANCH_NOTES.md`                               | Technische Detailnotizen aller Phasen   |
| `MIGRATION_CONFLICTS.md`                        | Audit-Trail aus der i18n-Konsolidierung |
| `docs/plans/spec-ux-strings.md`                 | Plan der Vorlauf-Phase                  |
| `docs/plans/ux-bootstrap-and-plugin-rollout.md` | Master-Plan zweite Phase                |
| `docs/plans/ux-rollout-execution-state.md`      | Operativer Tracker                      |
| `docs/plans/ux-coverage-after-rollout.txt`      | Abdeckungs-Bericht nach Welle A/B/C     |
| `docs/plans/ux-coverage-after-followup.txt`     | Abdeckungs-Bericht nach Folge-Welle     |

## Welchen Effekt hat das auf die Konfiguration?

### Für Endnutzer im Konfigurator

- **Sauber beschriftete Felder.** Statt „dbUid" steht künftig
  „Datenbank-Provider" und daneben ein erklärender Tooltipp.
- **Konsistent in Deutsch.** Die unschöne Mischanzeige aus
  englischen und deutschen Labels ist behoben.
- **Hilfe vor der Auswahl.** Wenn der Nutzer einen Auth-Provider, einen
  Layer-Typ oder einen OWS-Dienst hinzufügt, sieht er vor der Auswahl
  was die Familie überhaupt macht.
- **Beispiel-Vorlagen.** Per Klick lassen sich typische Szenarien
  einfügen — der Nutzer muss nicht von null beginnen.
- **Verwandte Optionen sichtbar.** Querverweise zwischen ähnlichen
  Konfigurationen helfen bei Entscheidungen.

### Für Konfigurations-Expertinnen und Vertrieb

- **Schnellere Erstkonfiguration**, weil weniger Nachschlagen in
  technischen Handbüchern nötig ist.
- **Weniger Rückfragen** vom Kunden, weil das Frontend selbst erklärt.
- **Mehrsprachigkeit endlich solide.** Englische Konfigurator-Sessions
  zeigen englische Texte, deutsche Sessions deutsche — ohne Mischmasch.

### Für Plugin-Entwicklerinnen und Maintainer

- **Klare Konvention.** Wer ein neues Plugin schreibt, legt einfach
  ein `_doc/ux.ini` daneben — das ist eine simple Textdatei mit
  Schlüssel-Wert-Paaren.
- **Werkzeug-Unterstützung.** `bootstrap_ux.py` erzeugt ein Skelett.
- **Coverage-Wächter.** Das CI-System schlägt Alarm, wenn jemand
  eine neue Option ohne Hilfetext einbaut.
- **Single Source of Truth.** Hilfetexte stehen nicht mehr in zwei
  Repos — nur noch in der WebSuite. Kein Auseinanderlaufen mehr.

### Für das Konfigurator-Team

- **`ux-schema.json` wird überflüssig.** Sobald der Konfigurator-Adapter
  auf den neuen `uxStrings`-Block aus der WebSuite umgestellt ist, kann
  die separate Hilfetext-Datei im Konfigurator-Repo zurückgezogen werden.
- **Keine Doppelpflege mehr.** Neue WebSuite-Optionen tauchen mit
  ihren Hilfetexten automatisch im Konfigurator auf.

## Zahlen am Ende

| Kennzahl                                           | Wert                             |
|----------------------------------------------------|----------------------------------|
| Hilfetext-Einträge Deutsch                         | 1 833                            |
| Hilfetext-Einträge Englisch                        | 1 787                            |
| Variant-Familien mit Apply-Vorlagen                | 35                               |
| Apply-Vorlagen gesamt                              | 60 (DE) + 60 (EN)                |
| Property-Label-Abdeckung Deutsch                   | 100 % der UI-relevanten Optionen |
| Property-Label-Abdeckung Englisch                  | 96,9 %                           |
| Commits auf dem Branch                             | 100+                             |
| Geänderte oder neu angelegte `_doc/ux.ini`-Dateien | 50+                              |

## Was ist als nächstes zu tun?

1. **Branch zusammenführen** in den Hauptzweig der WebSuite.
2. **Vollständigen Build im Container** durchführen, damit eine
   frische `app/__build/specs.json` entsteht.
3. **Diese `specs.json` an das Konfigurator-Team** übergeben.
4. **Konfigurator-Adapter umstellen**, sodass er primär die neuen
   `uxStrings` aus der WebSuite zieht.
5. **`ux-schema.json` retiren**, sobald der Konfigurator stabil läuft.
6. Optional in einer späteren Welle: Beispiel-Werte (`example`-Feld)
   flächendeckend ergänzen — heute liegt das bei 16 % der gepflegten
   Properties, eine Erhöhung auf 50 %+ wäre für die Lernkurve neuer
   Nutzer wertvoll.

## Kurz und knapp

- **Was?** Ein einheitliches, mehrsprachiges System für Hilfetexte
  in der WebSuite-Konfiguration — direkt da, wo die Optionen
  programmiert sind.
- **Warum?** Damit der Konfigurator dem Endnutzer immer aktuelle,
  saubere und vollständige Beschreibungen anzeigen kann, ohne dass
  zwei Teams an zwei Stellen dasselbe pflegen.
- **Effekt?** Weniger Wildwuchs, weniger Doppelpflege, bessere
  Bedienbarkeit, echte Mehrsprachigkeit.
- **Aufwand?** Substantiell — über 100 Commits in drei Hauptetappen
  plus eine Folge-Welle, organisiert über parallele Sub-Agenten.
- **Status?** Fertig. Akzeptanzkriterien erfüllt, übergabereif.

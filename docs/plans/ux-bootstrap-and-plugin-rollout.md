# Implementierungsplan: UX-Bootstrap-Tooling und Plugin-Pflege-Rollout

**Datum:** 2026-05-06
**Branch:** `feat/spec-ux-strings` (Folge-Arbeit nach Phase 1–10 dieses Branches)
**Vorlauf:** [`docs/plans/spec-ux-strings.md`](spec-ux-strings.md)
**Bezug:** [`konfigurator/docs/specs-generator-requirements.md`](../../../konfigurator/docs/specs-generator-requirements.md)

---

## Inhaltsverzeichnis

1. [Vision](#1-vision)
2. [Diagnose: heutiger Stand](#2-diagnose-heutiger-stand)
3. [Strategie-Übersicht](#3-strategie-übersicht)
4. [Phasen-Plan mit Parallelisierungs-Markierung](#4-phasen-plan-mit-parallelisierungs-markierung)
5. [Sub-Agent-Briefings (parallel ausführbar)](#5-sub-agent-briefings-parallel-ausführbar)
6. [Tooling-Spezifikationen](#6-tooling-spezifikationen)
7. [Spec-Erweiterung um `scenarios`](#7-spec-erweiterung-um-scenarios)
8. [Akzeptanzkriterien](#8-akzeptanzkriterien)
9. [Risiken und Annahmen](#9-risiken-und-annahmen)
10. [CI-Strategie (Diskussionsvorlage für Chef-Entwickler)](#10-ci-strategie-diskussionsvorlage-für-chef-entwickler)
11. [Übergabe](#11-übergabe)

---

## 1. Vision

Der GWS-Konfigurator soll für **alle Konfig-Properties und ‑Klassen** eine
qualitativ hochwertige, mehrsprachige UI-Hilfe rendern können — ohne eigene
hand-gepflegte `ux-schema.json`.

End-Zustand:

- Jede Konfigurations-Klasse, die der Konfigurator anzeigt, hat einen
  lesbaren `label`, einen `purpose`-Satz und ein `whenToUse`.
- Jede Property hat einen lesbaren `label` (statt `bindDN`/`tcLifeTime`)
  und einen kurzen `purpose`.
- Häufige Setup-Szenarien sind als applizierbare Templates verfügbar
  (`scenarios`-Block).
- Die Pflege ist **automatisiert vor-bestückt** durch ein Bootstrap-Tool,
  Maintainer poliert nur die Vorschläge.
- Die Coverage ist messbar (Coverage-Tool aus Phase 5 des Vorlauf-Branches)
  und steht als Audit-Werkzeug bereit. **CI-Gating wird in dieser Iteration
  nicht eingeführt**, sondern in einem späteren Schritt nach Abstimmung
  mit dem Chef-Entwickler (siehe §10).

## 2. Diagnose: heutiger Stand

Frische Messung mit `python -m gws.spec.generator.coverage --lang de`
(nach Abschluss der Vorlauf-Phasen):

### 2.1 Klassen-Doc-Coverage (`classDoc`)

| Lage                                    | Module |
|-----------------------------------------|---------|
| 100 % (vorbildlich)                     | `base.application.core`, `base.auth.{manager,method,mfa,provider,session_manager,sql_provider}`, `base.layer.{ows,tree}`, `base.legend.core`, `base.database.*`, `server.cli`, `server.core` |
| 50–80 % (akzeptabel)                    | `base.client.core`, `base.layer.core`, `base.model.core`, `base.metadata.core`, `qfieldcloud.action`, `qgis.cli`, `xml_helper` |
| 0–35 % (kritische Lücke)                | `base.action.cli`, `base.auth.cli`, `base.edit.api`, `base.edit.helper`, `base.exporter.action`, `base.map.action`, `qfieldcloud.cli`, `qfieldcloud.core`, `template.*`, `upload_helper` |

### 2.2 Property-Doc-Coverage (`propDoc`)

Stark schwankend. Vollständige Lücken:

- `base.edit.api` (0/20)
- `base.map.action` (0/20)
- `select_tool.action` (2/4)
- `qfieldcloud.cli` (0/4)
- `model.field` (10/18)
- `metadata.core` (43/82, also 40 fehlende Property-Docs)

### 2.3 UX-Strings-Coverage (`uxStrings.de`)

**Fast überall 0 %.** Einzige Ausnahme: `plugin/postgres` (~50 %, Pilot
aus Phase 7 des Vorlauf-Branches).

→ Hauptarbeit liegt hier. **Das ist der Schlüsselhebel** für die
Konfigurator-UI, weil ohne `uxStrings.label` keine Property in der UI
einen lesbaren Namen bekommt.

### 2.4 Konfigurator-Erwartung

Der HelpPanel (`konfigurator/.../HelpPanel.tsx`) konsumiert pro UID:

| Konfigurator-Feld | Spec-Quelle (heute) | Spec-Quelle (Ziel) |
|---|---|---|
| Lesbarer Name | clientseitiges `humanizeIdent(ident)` | `uxStrings[lang][uid].label` |
| Kurzbeschreibung | `strings[lang][uid]` | bleibt + zusätzlich `uxStrings.purpose` |
| Wann verwenden | nicht vorhanden | `uxStrings[lang][uid].whenToUse` |
| Komplexitäts-Filter | clientseitige Heuristik | `uxStrings[lang][uid].complexity` |
| Apply-Templates | hand-gepflegt in `ux-schema.json` | **`scenarios[lang][uid]`** (neu) |
| Beispielwert | nicht vorhanden | `uxStrings[lang][uid].example` |

→ Spec-Format-Erweiterung um `scenarios` ist Teil dieses Plans (siehe §7).

## 3. Strategie-Übersicht

Drei Stufen, jeweils mit klaren Vor-/Nach-Bedingungen für Parallelisierung:

```
Stufe 1: Tooling                                Sequentiell
─────────────────                              (1 Agent, 2 Phasen)
  • UX-Bootstrap-CLI bauen
  • scenarios-Spec-Erweiterung im Generator
                                                    ↓
                                            (Tools/Format ist nun da)
                                                    ↓
Stufe 2: Plugin-Pflege                          Parallel
──────────────────                              (3 Wellen × bis zu 4 Agenten)
  • Welle A — Auth/Login                            ↓
  • Welle B — Map/Project/Layer/Model               ↓
  • Welle C — Drucken/Templates/OWS/Edit            ↓
                                                    ↓
                                            (uxStrings + scenarios gefüllt)
                                                    ↓
Stufe 3: Konsolidierung                         Sequentiell
─────────────────────                          (1 Agent)
  • Coverage-Vergleich vor/nach
  • Übergabe an Konfigurator-Team
  • CI-Strategie als Diskussionsdokument
```

**Wichtiger Architektur-Punkt:** Sub-Agenten in Stufe 2 schreiben
ausschließlich neue Dateien (`_doc/ux.ini`, `_doc/scenarios.json`) in
disjunkten Plugin-Verzeichnissen. Damit gibt es **keine Datei-Konflikte**
zwischen parallel laufenden Agenten — kein lock, kein merge, keine
Race-Condition.

## 4. Phasen-Plan mit Parallelisierungs-Markierung

Legende: 🔒 sequentiell · 🟢 parallel ausführbar

### Phase 1 — UX-Bootstrap-CLI 🔒

**Output:** `app/gws/spec/generator/bootstrap_ux.py`

**Funktion:** liest `app/__build/specs.json` und erzeugt für ein Plugin
einen Vorschlags-Skelett `_doc/ux.ini` mit:

- `label` aus `Type.ident` per camelCase-Split + Domain-Wörterbuch
  (DN → DN, URL → URL, TC → TC, DPI → DPI, OGC → OGC, WMS/WMTS/WFS,
  PostGIS, …)
- `purpose` aus dem ersten Satz des Property-Docstrings (wenn vorhanden)
- `complexity` per Heuristik (`*Cache*`, `*Internal*`, `*Debug*`,
  `*Timeout*`, `*Concurrency*` → `advanced`; alles unter
  `host`/`port`/`url`/`user`/`password` → `basic`)

**CLI-Schnittstelle:**
```
python -m gws.spec.generator.bootstrap_ux <plugin-path> [--apply] [--lang de,en]
```

`--apply` ist **default off**: ohne diesen Flag wird der Vorschlag nur
nach stdout gerendert (Dry-Run). Mit `--apply` wird in
`<plugin-path>/_doc/ux.ini` geschrieben — aber nur, wenn die Datei nicht
existiert. Bestehende Dateien werden nie überschrieben.

**Tests:** in `app/gws/spec/generator/_test/bootstrap_ux_test.py`,
mindestens:
- `_label_from_ident('bindDN')` → `'Bind-DN'`
- `_label_from_ident('schemaCacheLifeTime')` → `'Schema-Cache-Lebensdauer'`
- `_label_from_ident('useCanvasExtent')` → `'Canvas-Ausdehnung verwenden'`
- Bootstrapping eines Fixture-Plugins → erzeugt `_doc/ux.ini` mit
  erwarteten Sektionen
- Idempotent: zweiter Lauf verändert nichts

**Commit-Bezeichnung:** `spec: ux bootstrap cli for label/purpose suggestions`

### Phase 2 — `scenarios`-Spec-Erweiterung 🔒

**Output:** Neuer Top-Level-Key `scenarios` in `specs.json`, neue Source-
Datei `_doc/scenarios.json` pro Plugin.

**Format `_doc/scenarios.json`:**
```json
{
  "gws.plugin.auth_provider.ldap.Config": [
    {
      "title": {"de": "Active Directory", "en": "Active Directory"},
      "purpose": {
        "de": "Standard-Setup für Windows-AD",
        "en": "Standard setup for Windows AD"
      },
      "template": {
        "url": "ldap://dc.example.com",
        "activeDirectory": true,
        "bindDN": "cn=gws-readonly,ou=services,dc=example,dc=com"
      }
    }
  ]
}
```

**Generator-Anbindung:**
- `app/gws/spec/generator/strings.py`: neue Funktion
  `collect_scenarios(gen)` — sammelt `_doc/scenarios.json` aus dem
  Source-Tree, validiert das Schema (title/purpose als lang-Map,
  template als JSON-Objekt), fasst nach `{lang: {uid: [scenarios]}}`
  zusammen.
- `app/gws/spec/core.py`: `SpecData.scenarios: dict`
- `app/gws/spec/generator/main.py`: in `_run_generator()` aufrufen,
  in `to_path()`/`from_path()` serialisieren
- `app/gws/spec/generator/typescript.py`: neuer Typ
  ```ts
  export interface UxScenario {
      title: string;
      purpose?: string;
      template: any;
  }
  export type UxScenarios = {[lang: string]: {[uid: string]: UxScenario[]}};
  ```

**Validierungsregeln:**
- Wenn `_doc/scenarios.json` syntaktisch ungültig → harter Fehler im
  Generator (kein Silent-Fail)
- Wenn UID im Spec nicht vorkommt → Warning im Build-Log, Eintrag
  trotzdem aufnehmen (UI-Konsument entscheidet)
- `template`-Objekt wird **nicht** gegen das Konfig-Schema validiert
  (das ist Sache des Konsumenten/Konfigurators)

**Tests:** drei pytest-Cases, neben den bestehenden:
- `_test/scenarios_test.py::test_collect_scenarios_parses_json`
- `_test/scenarios_test.py::test_invalid_json_raises`
- `_test/scenarios_test.py::test_unknown_uid_warns_but_keeps`

**Konventions-Doku:** `app/gws/spec/_doc/README.md` um den
`_doc/scenarios.json`-Abschnitt erweitern.

**Commit-Bezeichnung:** `spec: scenarios top-level key + per-plugin _doc/scenarios.json`

### Phase 3 — Plugin-Pflege Welle A: Auth/Login 🟢

**Parallel ausführbar mit bis zu 3 Sub-Agenten** (Aufteilung in §5).

**Plugins in dieser Welle:**

| # | Modul/Plugin | Begründung Welle A |
|---|---|---|
| 1 | `base/auth/manager` | Kern-Auth-Konfig, jeder Konfigurator-Nutzer sieht das |
| 2 | `base/auth/method` | Auth-Methoden-Konfiguration |
| 3 | `base/auth/mfa` | MFA-Konfig |
| 4 | `base/auth/provider` | Provider-Basis |
| 5 | `base/auth/session_manager` | Session-Konfig |
| 6 | `base/auth/sql_provider` | SQL-Auth-Provider |
| 7 | `plugin/auth_provider/ldap` | Häufigster Provider in Behörden-Setups |
| 8 | `plugin/auth_provider/file` | Standard-Fallback |
| 9 | `plugin/auth_method/{basic,token,web}` | Auth-Methoden-Plugins |
| 10 | `plugin/auth_mfa/{email,totp}` | MFA-Plugins |
| 11 | `plugin/auth_session_manager` | Session-Plugin |
| 12 | `plugin/account` | Self-Service-Account |

**Pro Plugin:**

1. `python -m gws.spec.generator.bootstrap_ux <plugin>` → Vorschlag prüfen
2. `python -m gws.spec.generator.bootstrap_ux <plugin> --apply` → schreiben
3. Manuelle Politur:
   - Englische `purpose`-Texte (1–3 Sätze pro Klasse)
   - Deutsche Übersetzung im `[de]`-Block
   - `whenToUse` für jede Klasse
   - `complexity` plausibilisieren
   - 1–3 `scenarios` pro Klasse, wenn das Plugin häufige Setup-Muster hat
4. `python -m gws.spec.generator.coverage --lang de` → Plugin sollte
   nahe 100 % uxStrings haben
5. Commit: `plugin/<name>: ux strings + scenarios`

**Akzeptanzkriterium pro Plugin:** uxStrings-Coverage ≥ 90 % im Coverage-Tool.

### Phase 4 — Plugin-Pflege Welle B: Map/Project/Layer/Model 🟢

**Parallel ausführbar mit bis zu 4 Sub-Agenten.**

| # | Modul/Plugin |
|---|---|
| 1 | `base/project` |
| 2 | `base/map` (`core`, `action`) |
| 3 | `base/layer` (`core`, `tree`, `group`, `ows`) |
| 4 | `base/model` (`core`, `field`, `default_model`, `scalar_field`, `validator`, `value`, `widget`) |
| 5 | `base/metadata.core` |
| 6 | `plugin/qgis` (alle Sub-Module) |
| 7 | `plugin/postgres.*` (Pilot existiert, scenarios ergänzen) |
| 8 | `plugin/geojson` |
| 9 | `plugin/tile_layer` |
| 10 | `plugin/mbtiles_layer` |
| 11 | `plugin/raster_layer` |
| 12 | `plugin/legend` |
| 13 | `plugin/dimension` |
| 14 | `plugin/model_field/*`, `plugin/model_widget/*`, `plugin/model_validator/*`, `plugin/model_value/*` |

**Pro Plugin:** identisches Vorgehen wie Welle A.

### Phase 5 — Plugin-Pflege Welle C: Drucken/Templates/OWS/Edit/Suche/Werkzeuge 🟢

**Parallel ausführbar mit bis zu 4 Sub-Agenten.**

| # | Modul/Plugin |
|---|---|
| 1 | `base/printer` |
| 2 | `base/template` |
| 3 | `plugin/template/{html,map,py,text}` |
| 4 | `base/exporter`, `plugin/exporter/*` |
| 5 | `plugin/csv_helper`, `plugin/upload_helper`, `plugin/email_helper` |
| 6 | `plugin/ows_server.{wms,wmts,wfs,csw}` |
| 7 | `plugin/ows_client.{wms,wmts,wfs}` |
| 8 | `base/edit` |
| 9 | `base/search` |
| 10 | `plugin/select_tool`, `plugin/identify_tool`, `plugin/location_tool`, `plugin/annotate_tool` |
| 11 | `plugin/storage_provider` |
| 12 | `plugin/nominatim`, `plugin/gbd_geoservices`, `plugin/gekos` |
| 13 | `plugin/alkis` (komplex, eigener Sub-Agent) |
| 14 | `plugin/qfieldcloud` |
| 15 | `plugin/xml_helper` |

**Pro Plugin:** identisches Vorgehen wie Welle A.

`plugin/alkis` ist groß und fachlich anspruchsvoll — bekommt einen
**eigenen Sub-Agenten**, der sich nicht parallel mit anderen ALKIS-
Themen kreuzt.

### Phase 6 — Konsolidierung 🔒

1. Coverage-Bericht erstellen:
   ```
   python -m gws.spec.generator.coverage --lang de > docs/plans/ux-coverage-after.txt
   diff <(coverage from before) docs/plans/ux-coverage-after.txt
   ```
2. `BRANCH_NOTES.md` aktualisieren — Welche Plugins sind nun gepflegt,
   welche Lücken bleiben
3. `MIGRATION_CONFLICTS.md` (aus Vorlauf-Branch Phase 6) auf Aktualität
   prüfen — eventuelle Orphans, die jetzt durch UX-Pflege geklärt wurden,
   markieren
4. Frische `specs.json` erzeugen, an Konfigurator-Team übergeben

**Commit-Bezeichnung:** `docs: ux pflege summary, coverage delta`

### Phase 7 — CI-Strategie als Diskussionsvorlage 🔒

**Wichtig:** *kein* Code, *kein* CI-Setup. Nur Dokumentation.

**Output:** `docs/plans/ci-coverage-gating.md` — eigenes Markdown-
Dokument als Vorbereitung für die Abstimmung mit dem Chef-Entwickler.
Inhalte:

- Vorgeschlagener Threshold-Fahrplan (start `30 %`, schrittweise auf
  `80 %`)
- Mock-CI-Snippet (auskommentiert) für GitHub Actions und GitLab CI
- Liste offener Fragen an den Chef-Entwickler:
  - Welcher CI-Provider wird genutzt?
  - Wie soll das Gate auf Releases vs. PRs greifen?
  - Wie werden Threshold-Erhöhungen mit Plugin-Maintainern kommuniziert?
  - Soll uxStrings-Coverage gleiches Gewicht haben wie classDoc/propDoc?
- Aufwandsschätzung für CI-Setup
- Risiken, wenn Threshold-Gate eingeführt wird (PR-Blocker bei neuen
  Properties)

**Commit-Bezeichnung:** `docs: ci-gating strategy proposal (for review)`

## 5. Sub-Agent-Briefings (parallel ausführbar)

Jeder Sub-Agent bekommt ein **eindeutig abgegrenztes Plugin-Set**, sodass
keine zwei Agenten dieselbe Datei schreiben. Vor Start jeder Welle wird
über `git status` / `git diff` geprüft, dass keine Plugin-Verzeichnisse
unfertige Änderungen haben.

### Welle A: Auth/Login (3 Sub-Agenten parallel)

#### Agent A1 — Auth-Kern-Klassen
**Verantwortung:**
- `app/gws/base/auth/manager/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/method/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/provider/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/session_manager/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/sql_provider/_doc/{ux.ini,scenarios.json}`

**Anti-Verantwortung:** Plugin-Provider (`auth_provider/*`), MFA, Account.

#### Agent A2 — MFA + Auth-Methoden-Plugins
**Verantwortung:**
- `app/gws/base/auth/mfa/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_method/{basic,token,web}/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_mfa/{email,totp}/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_session_manager/_doc/{ux.ini,scenarios.json}`

**Anti-Verantwortung:** Auth-Kern, Provider-Plugins, Account.

#### Agent A3 — Auth-Provider-Plugins + Account
**Verantwortung:**
- `app/gws/plugin/auth_provider/file/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_provider/ldap/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/account/_doc/{ux.ini,scenarios.json}`

**Anti-Verantwortung:** Auth-Kern, MFA, Methoden-Plugins.

### Welle B: Map/Project/Layer/Model (4 Sub-Agenten parallel)

#### Agent B1 — Project + Map
**Verantwortung:**
- `app/gws/base/project/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/map/_doc/{ux.ini,scenarios.json}` (für `core` und `action`)
- `app/gws/base/metadata/_doc/{ux.ini,scenarios.json}`

#### Agent B2 — Layer-Kern
**Verantwortung:**
- `app/gws/base/layer/_doc/{ux.ini,scenarios.json}` (deckt `core`, `tree`, `group`, `ows`)
- `app/gws/plugin/legend/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/dimension/_doc/{ux.ini,scenarios.json}`

#### Agent B3 — Layer-Plugins
**Verantwortung:**
- `app/gws/plugin/qgis/_doc/{ux.ini,scenarios.json}` (gilt für alle Sub-Module)
- `app/gws/plugin/postgres/_doc/scenarios.json` (ux.ini schon vorhanden — nur scenarios)
- `app/gws/plugin/geojson/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/tile_layer/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/mbtiles_layer/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/raster_layer/_doc/{ux.ini,scenarios.json}`

#### Agent B4 — Model-Welt
**Verantwortung:**
- `app/gws/base/model/_doc/{ux.ini,scenarios.json}` (deckt alle Submodule)
- `app/gws/plugin/model_field/_doc/{ux.ini,scenarios.json}` (deckt alle Field-Typen)
- `app/gws/plugin/model_widget/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/model_validator/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/model_value/_doc/{ux.ini,scenarios.json}`

### Welle C: Drucken/Templates/OWS/Edit/Suche/Werkzeuge (4 Sub-Agenten parallel)

#### Agent C1 — Drucken + Templates + Export
**Verantwortung:**
- `app/gws/base/printer/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/template/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/template/_doc/{ux.ini,scenarios.json}` (deckt html/map/py/text)
- `app/gws/base/exporter/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/exporter/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/{csv,upload,email}_helper/_doc/{ux.ini,scenarios.json}`

#### Agent C2 — OWS (Server + Client)
**Verantwortung:**
- `app/gws/plugin/ows_server/_doc/{ux.ini,scenarios.json}` (deckt wms/wmts/wfs/csw)
- `app/gws/plugin/ows_client/_doc/{ux.ini,scenarios.json}` (deckt wms/wmts/wfs)
- `app/gws/plugin/xml_helper/_doc/{ux.ini,scenarios.json}`

#### Agent C3 — Edit/Suche/Werkzeuge
**Verantwortung:**
- `app/gws/base/edit/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/search/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/select_tool/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/identify_tool/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/location_tool/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/annotate_tool/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/storage_provider/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/nominatim/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/gbd_geoservices/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/gekos/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/qfieldcloud/_doc/{ux.ini,scenarios.json}`

#### Agent C4 — ALKIS (Solo)
**Verantwortung:**
- `app/gws/plugin/alkis/_doc/{ux.ini,scenarios.json}` (komplexes
  Liegenschaftskataster-Plugin, eigener Agent wegen Fachlichkeit)

### Standard-Briefing-Vorlage für Sub-Agenten

Jeder Sub-Agent bekommt einen Prompt nach folgendem Schema:

```
Aufgabe: UX-Strings für Plugin(s) <X, Y, Z> im WebSuite-Repo pflegen.

Branch: feat/spec-ux-strings (bereits ausgecheckt — bleibe drauf, nicht
checkout)

Verantwortung: NUR die unter "Verantwortung" gelisteten Dateipfade
schreiben. Andere Pfade nicht anfassen.

Vorgehen:
1. ./make.sh spec   (sicherstellen, dass __build/specs.json frisch ist)
2. Pro Plugin <P>:
   a. python -m gws.spec.generator.bootstrap_ux <P> → Vorschlag prüfen
   b. python -m gws.spec.generator.bootstrap_ux <P> --apply
   c. Polierung (siehe Qualitätskriterien unten)
   d. python -m gws.spec.generator.coverage --lang de | grep <P>
      → uxStrings ≥ 90 % erreichen
   e. Optional: _doc/scenarios.json für 1–3 typische Setup-Muster
   f. git add app/gws/<P>/_doc/ && git commit -m "..."

Qualitätskriterien:
- label: 1–4 Wörter, kein Variablen-Camelcase, deutscher Begriff
- purpose: 1–3 Sätze, fachlich korrekt, allgemein verständlich
- whenToUse: nur wenn substantiell ergänzend; lieber leer als banal
- complexity: basic (Anfänger), intermediate (Standard), advanced (Power-User)
- example: nur wo der Wert nicht offensichtlich ist
- Kein Phantasietext: wenn Quelle dünn, Lücken lassen

Commit-Konvention:
  plugin/<name>: ux strings (welle <A|B|C>)

Begrenzung:
- Generator-Code nicht ändern
- Andere Plugins nicht anfassen
- Keine Tests im Generator-Test-Pfad ändern
- Keine Konfigurator-Repo-Dateien
```

## 6. Tooling-Spezifikationen

### 6.1 `bootstrap_ux.py` — Detail

**Heuristik für `_label_from_ident()`:**

```python
DOMAIN_TERMS_DE = {
    'dn': 'DN',
    'url': 'URL',
    'uri': 'URI',
    'tcp': 'TCP',
    'ip': 'IP',
    'dpi': 'DPI',
    'ogc': 'OGC',
    'wms': 'WMS', 'wmts': 'WMTS', 'wfs': 'WFS', 'csw': 'CSW',
    'sql': 'SQL', 'crs': 'CRS', 'epsg': 'EPSG',
    'json': 'JSON', 'xml': 'XML', 'csv': 'CSV',
    'tc': 'TC', 'mfa': 'MFA', 'totp': 'TOTP',
    'pdf': 'PDF', 'png': 'PNG', 'tiff': 'TIFF',
    'postgis': 'PostGIS', 'qgis': 'QGIS', 'alkis': 'ALKIS',
}

VERB_PREFIXES = {
    'use':  'verwenden',
    'is':   'ist',
    'has':  'hat',
    'allow': 'erlaubt',
    'enable': 'aktiviert',
    'disable': 'deaktiviert',
}
```

**Algorithmus:**
1. camelCase splitten: `schemaCacheLifeTime` → `['schema', 'cache', 'life', 'time']`
2. Jeden Teil:
   - in `DOMAIN_TERMS_DE` nachschlagen → ersetze
   - sonst capitalize
3. Wenn erstes Wort in `VERB_PREFIXES`: `useCanvasExtent` → "Canvas-Ausdehnung verwenden" (Verb hinten)
4. Zusammenfügen mit `-` oder Leerzeichen

**Tests:**
- 12 erwartete (ident, label_de)-Pairs als parametrisierte Tests
- Roundtrip-Test: für ein bestehendes Plugin (postgres, das in
  uxStrings.de bereits Labels hat) sollten die Bootstrap-Vorschläge
  überlappen

**Empfehlung:** `bootstrap_ux.py` schreibt `_doc/ux.ini` mit
**vorangestelltem Kommentar**, dass Texte vom Tool generiert wurden:

```ini
# Auto-generated skeleton from bootstrap_ux.py.
# Bitte Texte überprüfen, anpassen, deutsche purpose-Texte ergänzen.

[de]
…
```

So weiß der nächste Maintainer, dass die Datei ggf. noch nicht
production-quality ist.

### 6.2 Coverage-Tool-Erweiterung (klein)

`coverage.py` braucht eine neue Option:
```
--per-plugin <name>     # Nur ein Plugin anzeigen, für Sub-Agent-Verifikation
--out json              # JSON-Output statt ASCII-Tabelle, für Tooling
```

Optional, kann auch in einer separaten Mini-Phase implementiert werden,
falls die Sub-Agenten einen reineren Coverage-Output für ihre
Verifikation brauchen.

### 6.3 Verzicht auf `--auto-apply`

Bewusst: Das Tool schreibt **keine produktiven Texte** ohne menschlichen
Review. Die Idee ist Bootstrapping, nicht Auto-Pilot. Wenn ein
Maintainer 50 Plugins hätte, würde er sie sequentiell durchgehen, oder
ein eigenes Skript mit `--apply` über alle laufen lassen — das ist OK,
aber außerhalb der Tool-Verantwortung.

## 7. Spec-Erweiterung um `scenarios`

### 7.1 Format-Entscheidung: JSON statt INI

Begründung gegen INI: `template`-Werte sind beliebig geschachtelte
JSON-Strukturen (`{"url": "...", "options": {"ssl": true}}`). Das in INI
zu serialisieren ist umständlich (escape, multi-line). JSON ist:

- Strukturell, einfacher zu validieren
- Gleiche Notation wie der Konfigurator sie konsumiert (er rendert ein
  JSON-Objekt als Apply-Template direkt)
- Weniger Format-Drift zwischen Source und UI

### 7.2 Datei-Layout

Pro Plugin **eine** Datei `_doc/scenarios.json` mit allen UID-Einträgen.
Beispiel:

```json
{
  "gws.plugin.auth_provider.ldap.Config": [
    {
      "title": {"de": "Active Directory", "en": "Active Directory"},
      "purpose": {
        "de": "Standard-Setup für eine Microsoft-AD-Anbindung mit gws-readonly Service-Account.",
        "en": "Standard setup for a Microsoft AD connection with a gws-readonly service account."
      },
      "template": {
        "url": "ldap://dc.example.com",
        "activeDirectory": true,
        "bindDN": "cn=gws-readonly,ou=services,dc=example,dc=com",
        "displayNameFormat": "{givenName} {sn}"
      }
    },
    {
      "title": {"de": "OpenLDAP, anonyme Suche", "en": "OpenLDAP, anonymous search"},
      "template": {
        "url": "ldap://ldap.example.com",
        "activeDirectory": false
      }
    }
  ],
  "gws.plugin.auth_provider.ldap.UserSpec": [
    {
      "title": {"de": "Behördennutzer-Gruppe", "en": "Public-sector group"},
      "template": {
        "memberOf": "cn=gws-users,ou=groups,dc=example,dc=com",
        "roles": ["all_users", "edit"]
      }
    }
  ]
}
```

### 7.3 Spec-Output

Top-Level-Block in `specs.json`:
```json
{
  "scenarios": {
    "de": {
      "gws.plugin.auth_provider.ldap.Config": [
        {"title": "Active Directory", "purpose": "…", "template": {…}}
      ]
    },
    "en": { … }
  }
}
```

Der Sammler in `strings.collect_scenarios()` faltet die `lang`-Maps in
`title`/`purpose` zur Laufzeit auf.

### 7.4 Backwards-Compat

`from_path()` setzt fehlendes `scenarios` auf `{}`. Konfigurator behandelt
fehlenden Block als „keine Szenarien für diese UID" und fällt auf den
heutigen `ux-schema.json`-Adapter zurück (siehe Vorlauf-Branch
`BRANCH_NOTES.md`).

## 8. Akzeptanzkriterien

Stufenweise pro Phase definiert:

### Nach Phase 1 (Tooling)
1. `python -m gws.spec.generator.bootstrap_ux --help` läuft
2. Tool produziert für ein Test-Plugin eine valide `ux.ini` als Vorschlag
3. Bestehende `ux.ini` wird nie überschrieben
4. Tests in `_test/bootstrap_ux_test.py` grün

### Nach Phase 2 (scenarios)
5. `specs.json` hat Top-Level-Key `scenarios`
6. `gws.generated.ts` typisiert `UxScenario` und `UxScenarios`
7. Backwards-Compat: alte `specs.json` ohne `scenarios` lädt
8. Konventions-Doku in `app/gws/spec/_doc/README.md` ergänzt

### Nach Phase 3–5 (Plugin-Pflege)
9. Coverage `uxStrings.de` ≥ 70 % über alle gepflegten Module
10. Mindestens 3 Plugins haben `_doc/scenarios.json` mit ≥ 2 Szenarien
11. Pro Plugin ein Commit mit klarem Scope
12. `make.sh spec` läuft ohne Warnings (außer denen aus
    `MIGRATION_CONFLICTS.md` aus dem Vorlauf-Branch)

### Nach Phase 6 (Konsolidierung)
13. Coverage-Delta-Bericht als Markdown im Branch
14. `BRANCH_NOTES.md` aktualisiert mit Status der UX-Pflege

### Nach Phase 7 (CI-Doku)
15. `docs/plans/ci-coverage-gating.md` existiert
16. Enthält ≥ 5 konkrete Fragen für den Chef-Entwickler
17. Enthält **keinen** ausführbaren CI-Code (auskommentiert ist OK)

## 9. Risiken und Annahmen

| Risiko | Wahrscheinlichkeit | Mitigation |
|---|---|---|
| Bootstrap-Tool generiert schlechte Labels | mittel | `_label_from_ident()` ist Vorschlags-Generator, **nicht** Auto-Apply. Maintainer reviewt zwingend. |
| Sub-Agenten schreiben unbeabsichtigt in fremde Pfade | gering | Briefing pro Agent listet exakte Verantwortungs-Pfade; `git diff` vor Commit prüfen |
| Parallele Branches divergieren | gering | Alle Sub-Agenten arbeiten **im selben Branch** auf disjunkten Dateien; kein Merge nötig |
| Generator-Erweiterung in Phase 2 bricht bestehenden Build | gering | Tests in `_test/scenarios_test.py`, Backwards-Compat-Check, lokaler `./make.sh spec` als Smoke |
| Pflegetexte fachlich falsch | mittel | Maintainer reviewt; keine Phantasie aus dem Tool; Lücken explizit erlaubt |
| `scenarios.json` Schema driftet | gering | Generator validiert hartes Schema; harter Fehler bei Verstoß |
| Sub-Agent-Briefing wird ignoriert | gering | Briefing endet mit klaren „nicht anfassen"-Listen; Code-Review beim Merge der Wellen |
| Coverage-Tool-Output bei vielen Modulen unübersichtlich | gering | `--per-plugin` Filter (Phase 6.2); JSON-Output für Tooling |

### Annahmen (offene Punkte für Diskussion)

- **Annahme A1:** Texte werden auf Deutsch und Englisch gepflegt.
  Andere Sprachen (z. B. Französisch für ALKIS-Nachbarregionen) sind
  out-of-scope.
- **Annahme A2:** Das Konfigurator-Team ist bereit, den
  `scenarios`-Block zu konsumieren. Falls nein → Phase 2 verschiebbar,
  Plugin-Pflege geht trotzdem mit `ux.ini` voran.
- **Annahme A3:** Plugin-Maintainer haben Zeit, manuell zu polieren —
  oder der ausführende Agent darf Texte aus existierenden Docstrings
  und `_doc/strings.ini` ableiten ohne weiteren Review.

## 10. CI-Strategie (Diskussionsvorlage für Chef-Entwickler)

**Status: Nicht umsetzen, nur dokumentieren.** Ergebnis dieser Phase ist
ausschließlich `docs/plans/ci-coverage-gating.md` als Diskussionsvorlage.

### 10.1 Ziel

Verhindern, dass die mühsam aufgebaute UX-Coverage durch unbedacht
hinzugefügte Properties wieder verfällt. Ein CI-Gate, das auf
Pull-Requests gegen `master` greift, würde:

- den Coverage-Lauf ausführen
- bei Threshold-Verletzung den PR rot markieren
- Maintainer zwingen, neue Properties mit `_doc/ux.ini`-Eintrag
  einzuchecken

### 10.2 Vorgeschlagener Threshold-Fahrplan

```
Monat 1 (nach Plugin-Welle A):  --threshold 30  (uxStrings)
Monat 2 (nach Plugin-Welle B):  --threshold 50
Monat 3 (nach Plugin-Welle C):  --threshold 70
Monat 4 (Konsolidierung):       --threshold 80
Stabilstand:                    --threshold 80 dauerhaft
```

Erst hochziehen, wenn die jeweilige Welle abgeschlossen ist — sonst
blockiert das Gate die Pflege selbst.

### 10.3 Mock-CI-Snippets (zur Diskussion, nicht aktivieren)

**GitHub Actions:**
```yaml
# .github/workflows/spec-coverage.yml — DRAFT, NOT ACTIVE
# name: spec-coverage
# on:
#   pull_request:
#     paths:
#       - 'app/**'
# jobs:
#   coverage:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: { python-version: '3.12' }
#       - run: pip install -r install/pip.lst
#       - run: ./make.sh spec
#       - run: python -m gws.spec.generator.coverage --lang de --threshold 30
```

**GitLab CI:**
```yaml
# .gitlab-ci.yml fragment — DRAFT, NOT ACTIVE
# spec-coverage:
#   image: gbdconsult/gws-amd64:8.4
#   script:
#     - ./make.sh spec
#     - python -m gws.spec.generator.coverage --lang de --threshold 30
#   only:
#     - merge_requests
```

### 10.4 Offene Fragen an den Chef-Entwickler

1. Welcher CI-Provider wird tatsächlich verwendet (GitHub Actions,
   GitLab CI, Jenkins, anderer)?
2. Soll das Gate **nur auf Pull-Requests** greifen oder auch auf
   `master`-Pushes?
3. Wie werden Threshold-Erhöhungen kommuniziert? Soll ein Maintainer-
   Channel (Slack, Mailingliste) informiert werden, bevor der
   Threshold steigt?
4. Sollen `classDoc` / `propDoc` / `uxStrings` **getrennt** gegated
   werden, oder reicht eine Kombi-Metrik?
5. Wie umgehen mit Plugins, die explizit unfertig sind
   (z. B. `qfieldcloud` als Beta)? Whitelist oder pro-Plugin-Threshold?
6. Soll die Coverage-Verschlechterung (Δ < 0) ein Signal sein, auch
   wenn der absolute Threshold erfüllt ist?
7. Wer ist Owner für Folge-Diskussionen, wenn ein PR an der Coverage
   scheitert?
8. Aufwand: Setup einmalig ~2 h, Wartung minimal — wer übernimmt das?
9. Soll `scenarios`-Coverage Teil des Gates sein, oder nur das
   `uxStrings.label`/`purpose`-Pflicht-Subset?

### 10.5 Alternative ohne CI-Gate

Falls der Chef-Entwickler kein CI-Gate möchte (zu viele PR-Blocker),
gibt es Alternativen:

- **Nur Reporting**: Coverage-Tool wird nightly gegen `master`
  ausgeführt, Ergebnis als Markdown-Bericht in das Repo committed.
  Sichtbar, aber blockiert nicht.
- **Pre-Merge-Check**: kein hartes Gate, aber GitHub-Bot hinterlässt
  einen Coverage-Kommentar im PR
- **Documentation-Only**: das Coverage-Tool bleibt als Werkzeug
  verfügbar, Maintainer-Konvention regelt die Nutzung

## 11. Übergabe

Wenn alle Phasen abgeschlossen sind:

1. **`BRANCH_NOTES.md`** im Repo-Root um eine Sektion „UX-Pflege-
   Status" erweitern: pro Plugin grüner/gelber/roter Status (Coverage-
   Schwellen).
2. **Frische `specs.json`** aus dem Branch an das Konfigurator-Team
   senden — diese Datei sollte `uxStrings`-Abdeckung > 70 % und einen
   ersten Satz `scenarios` enthalten.
3. **Konfigurator-Team** kann seinen `getUxEntry()`-Helper auf das
   neue Format umstellen und schrittweise `ux-schema.json` retiren.
4. **`docs/plans/ci-coverage-gating.md`** an den Chef-Entwickler
   senden, Termin für Entscheidung über CI-Gating einplanen.

### Erwarteter Branch-Stand am Ende

```
feat/spec-ux-strings  (gleicher Branch)
├── (alle bestehenden Vorlauf-Commits aus spec-ux-strings.md)
├── docs: implementation plan for ux-bootstrap-and-plugin-rollout
├── spec: ux bootstrap cli for label/purpose suggestions
├── spec: scenarios top-level key + per-plugin _doc/scenarios.json
├── plugin/<welle-A-plugins>: ux strings (n Commits parallel)
├── plugin/<welle-B-plugins>: ux strings (n Commits parallel)
├── plugin/<welle-C-plugins>: ux strings (n Commits parallel)
├── docs: ux pflege summary, coverage delta
└── docs: ci-gating strategy proposal (for review)
```

Etwa **50–80 Commits zusätzlich**, je nach Plugin-Granularität. Im
Branch wirken sie als Sequenz; durch parallele Sub-Agenten innerhalb
einer Welle ist der Wallclock-Aufwand jedoch deutlich kürzer als die
Commit-Zahl suggeriert.

---

## Anhang: Glossar

- **Sub-Agent** — eine Claude-Code-Agent-Instanz, die parallel zu
  anderen Agenten ein klar abgegrenztes Set von Plugin-Verzeichnissen
  pflegt. Verwendet dieselbe Branch-Working-Copy, schreibt aber nur in
  disjunkte Pfade.
- **Welle** — eine Gruppe parallel ausführbarer Plugin-Pflege-Tasks,
  zusammengefasst nach Funktions-Cluster (Auth, Karte, Drucken, …).
  Wellen laufen sequentiell hintereinander, aber innerhalb einer Welle
  arbeiten Sub-Agenten parallel.
- **Bootstrap** — automatisierte Erst-Befüllung von `_doc/ux.ini` mit
  Vorschlägen aus dem Spec. Maintainer reviewt und poliert.
- **Pareto-Plugin** — ein Plugin, das im Konfigurator-UI sehr häufig
  angezeigt wird und daher Priorität in der Pflege bekommt.

## Anhang: Referenzen

- Vorlauf-Branch: dieser Branch (`feat/spec-ux-strings`),
  Commits `8818d48b` bis `a3cce3d8`.
- Anforderungsdokument:
  [`konfigurator/docs/specs-generator-requirements.md`](../../../konfigurator/docs/specs-generator-requirements.md)
- Vorlauf-Plan:
  [`docs/plans/spec-ux-strings.md`](spec-ux-strings.md)
- Konvention:
  [`app/gws/spec/_doc/README.md`](../../app/gws/spec/_doc/README.md)
- Coverage-Tool: `python -m gws.spec.generator.coverage --help`
- Konfigurator-Konsumstelle:
  `konfigurator/GWS-Konfigurator/src/components/help/HelpPanel.tsx`,
  `src/lib/uxSchema.ts`

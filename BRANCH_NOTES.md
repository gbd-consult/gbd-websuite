# Branch Notes — `feat/spec-ux-strings`

**Branch:** `feat/spec-ux-strings` (von `master`)
**Anforderungsdokument:** `konfigurator/docs/specs-generator-requirements.md`
**Implementierungspläne:**
- [`docs/plans/spec-ux-strings.md`](docs/plans/spec-ux-strings.md) (Vorlauf-Phase)
- [`docs/plans/ux-bootstrap-and-plugin-rollout.md`](docs/plans/ux-bootstrap-and-plugin-rollout.md) (Master-Plan zweite Phase)
- [`docs/plans/ux-rollout-execution-state.md`](docs/plans/ux-rollout-execution-state.md) (Operativer Tracker)

---

## TL;DR

Der Spec-Generator schreibt jetzt einen neuen Top-Level-Block
`uxStrings` in `specs.json` und in `gws.generated.ts`. Der Block enthält
strukturierte UX-Doku (label, purpose, whenToUse, complexity, …) pro
Klasse und Property, gesammelt aus `_doc/ux.ini`-Dateien sowie aus
optionalen Field-List-Markern im Klassen-Docstring.

`plugin/postgres` ist als Pilot vollständig auf das neue Format
umgestellt. Die Konvention ist unter `app/gws/spec/_doc/README.md`
dokumentiert.

Frische `specs.json` ohne `uxStrings`-Key bleibt rückwärtskompatibel:
`from_path()` setzt einen leeren Default.

## Was wurde geändert?

| Bereich | Datei(en) | Inhalt |
|---|---|---|
| Spec-Format | `app/gws/spec/core.py` | `SpecData.uxStrings` |
| Generator-Pipeline | `app/gws/spec/generator/main.py`, `base.py`, `strings.py` | Neue Sammler `collect_docstring_markers`, `collect_ux`, `apply_ux_to_variants`. Pipeline-Reihenfolge: marker → ux.ini → variant-backfill → strings.collect → typescript. |
| Tests | `app/gws/spec/generator/_test/` (neu) | 8 pytest-Cases, ohne Docker lauffähig (`PYTHONPATH=app pytest app/gws/spec/generator/_test/`). |
| TypeScript | `app/gws/spec/generator/typescript.py` | Exportiert `UxEntry` und `UxStrings`-Typ. |
| Coverage-CLI | `app/gws/spec/generator/coverage.py` (neu) | `python -m gws.spec.generator.coverage`. |
| i18n-Migration | `app/gws/spec/strings.de.ini.alt` (gelöscht) → 10 × `_doc/strings.ini` | 145 Einträge in Modul-Dateien gemerged. Konflikte und Orphans in `MIGRATION_CONFLICTS.md`. |
| Pilot | `app/gws/plugin/postgres/_doc/ux.ini` (neu) | Provider, finder, layer, model, auth_provider mit DE+EN. |
| Configref | `app/gws/spec/generator/configref.py` | UX-Block pro Klasse, vor der Property-Tabelle. |
| Konvention | `app/gws/spec/_doc/README.md` (neu) | Format-Doku für `_doc/strings.ini` und `_doc/ux.ini`. |

## Neue CLI-Schalter

```
python -m gws.spec.generator.coverage [--lang de] [--threshold 80] [--manifest <path>]
```

`--threshold N` lässt den Tool-Lauf mit Exit-Code 1 abbrechen, sobald
irgendeine Metrik (classDoc, propDoc, uxStrings) in irgendeinem Modul
unter `N` Prozent fällt — taugt als CI-Guard gegen neu eingefügte
undokumentierte Konfig-Properties.

## Neue Source-Konventionen für Plugin-Maintainer

1. **`_doc/ux.ini`** pro Modul für strukturierte UX-Doku — siehe
   `app/gws/spec/_doc/README.md`.
2. **Field-List-Marker im Klassen-Docstring** (optional):
   ```python
   class Config(gws.Config):
       """LDAP auth provider config.

       :complexity: intermediate
       :seeAlso: gws.plugin.auth_provider.file.Config
       """
   ```
   Erkannte Marker: `:complexity:`, `:seeAlso:`, `:since:`, `:deprecated:`.
3. **Vorrangregel:** `_doc/ux.ini` > Docstring-Marker.
4. Backwards-compatibel: alle bisherigen Plugins bleiben nutzbar, der
   `uxStrings`-Block ist im JSON einfach leer für Module ohne `ux.ini`.

## i18n-Konsolidierung (Phase 6) — bewusste Designentscheidungen

Statistik des Migrationslaufs:

| Klasse                       | Anzahl |
|------------------------------|--------|
| identisch (übersprungen)     | 16     |
| in Modul-Datei gemerged      | 145    |
| Konflikt (Modul-Text behalten) | 32   |
| Orphans (kein Modul-Owner)   | 384    |

**Konflikt-Strategie:** wenn dieselbe UID sowohl in `strings.de.ini.alt`
als auch in einer Modul-`_doc/strings.ini` einen DE-Wert hatte, **wird
die Modul-Version behalten**. Der `.alt`-Wert wird nicht überschrieben,
sondern in [`MIGRATION_CONFLICTS.md`](MIGRATION_CONFLICTS.md) zur
manuellen Sichtung gesammelt. Begründung: die Modul-Texte sind
durchweg ausführlicher und neuer.

**Orphans:** UIDs aus `.alt`, deren Präfix kein heutiges Modul matcht
(meist `cli.*` und `gws.auth.types.*` — alte Strukturen, die der
Codebase entwachsen sind). Keine automatische Zuordnung möglich, daher
ebenfalls in `MIGRATION_CONFLICTS.md`. Maintainer kann Orphans
selektiv reaktivieren, indem er die UIDs in das passende
`_doc/strings.ini` einträgt.

## Configref-Renderer (Phase 8) — bewusste Designentscheidungen

Die Anforderung verlangt UX-Felder im `configref.md`. Das
out-of-scope-Kapitel der Anforderung erlaubt aber explizit, **keinen
großen Renderer-Umbau** für mehrzeilige Tabellenzellen zu machen.

Pragmatischer Patch:

- UX-Felder werden **als Markdown-Block oberhalb der Property-Tabelle**
  pro Klasse gerendert, nicht in den Tabellenzellen.
- Mehrzeiliges `purpose` wird eingerückt statt escaped.
- Property-Ebene bekommt **keinen** UX-Block (das wäre der größere
  Renderer-Umbau).

Wenn später ein vollständiger Renderer-Umbau ansteht, ist das eine
isolierte Folgearbeit. Für den Konfigurator-Konsumenten zählt sowieso
`specs.json`, nicht die `configref.md`.

## Tests

```
PYTHONPATH=app pytest app/gws/spec/generator/_test/   # 8 Cases, läuft ohne Docker
./make.sh test go -k ux                               # vollständig im Container-Stack
```

## UX-Pflege-Status (zweite Phase, Welle A/B/C)

Nach dem Pilot postgres wurden in 3 parallelisierten Wellen alle
WebSuite-Plugins und Base-Module mit `_doc/ux.ini` versehen. Pro Welle
wurden 3-5 Sub-Agenten parallel gespawnt; jeder Sub-Agent erzeugte mit
`bootstrap_ux.py` ein Skelett, polierte zu echtem Deutsch und
committete pro Plugin separat.

**Tooling, das in dieser Phase entstanden ist:**

- `app/gws/spec/generator/bootstrap_ux.py` — CLI, das aus `specs.json`
  ein `_doc/ux.ini`-Skelett mit Label-/Purpose-/Complexity-Vorschlägen
  erzeugt. Tests in `_test/bootstrap_ux_test.py`.
- `app/gws/spec/generator/strings.py::collect_scenarios` — neuer
  Sammler für `_doc/scenarios.json` (Apply-Templates pro UID).
  Top-Level-Key `scenarios` in `specs.json`. Tests in
  `_test/scenarios_test.py`.

**Coverage nach Rollout** (Stand: nach Welle C, frische `specs.json`):

| | |
|---|---|
| `uxStrings.de` Einträge gesamt | 838 |
| `uxStrings.en` Einträge gesamt | 792 |
| `scenarios.de` UIDs | 28 |
| `scenarios` Apply-Templates gesamt | 38 (de) + 38 (en) |

Per-Modul-Bericht in [`docs/plans/ux-coverage-after-rollout.txt`](docs/plans/ux-coverage-after-rollout.txt).

**Vollständig gepflegte Bereiche:**

- Welle A — Auth/Login: `gws.base.auth` + `auth_method`, `auth_mfa`,
  `auth_provider`, `auth_session_manager`, `account` (alle Configs ≥ 100 %
  uxStrings.de).
- Welle B — Map/Project/Layer/Model: `base.{project,map,layer,model,
  metadata}` plus alle Layer-Plugins (`qgis`, `geojson`, `tile_layer`,
  `mbtiles_layer`, `raster_layer`) und Model-Plugins (`model_field`,
  `model_widget`, `model_validator`, `model_value`).
- Welle C — Drucken/Templates/OWS/Edit/Werkzeuge/ALKIS:
  `base.{printer,template,exporter,edit,search}` plus alle entsprechenden
  Plugins (`ows_server`, `ows_client`, `xml_helper`, `template`,
  `exporter`, `csv_helper`, `upload_helper`, `email_helper`,
  `select_tool`, `identify_tool`, `location_tool`, `annotate_tool`,
  `storage_provider`, `nominatim`, `gbd_geoservices`, `gekos`,
  `qfieldcloud`, `alkis`).

**Ausgelassene Bereiche** (bewusst, kein UX-Konsument):

- `gws` Top-Level (Core-Types, Mixins).
- `gws.lib.*`, `gws.gis.*` (interne Helfer).
- `gws.base.{client,database,web,application,storage}.*` (Infrastruktur).
- `gws.base.action.cli`, `*.api`, `*.cli` (interne Wire-Schemas).
- `*.Object`, `*.Props` in vielen Plugins (Runtime-Repräsentationen).

**Beobachtete Auffälligkeiten** (Sub-Agent-Berichte, nicht im Branch
gefixt — siehe Konsolidierungsbericht für Details):

- `bootstrap_ux.py` schlägt für rein-englische Identifier (`Url`,
  `Time`) automatisch deutsche Labels vor, die manuell gepflegt werden
  müssen (z.B. `Url` → `URL`). Die Heuristik sieht das als Default-
  Vorschlag; Maintainer überschreiben händisch.
- Mehrere Module zeigen niedrige `classDoc`-Coverage (Object-Klassen
  ohne Docstring). UX-Coverage ist davon entkoppelt — uxStrings deckt
  diese ab.

## Folge-Welle (Konfigurator-Anforderung A1–A6)

Nach Welle A/B/C kam vom Konfigurator-Team eine zweite Anforderungs-
runde: `konfigurator/docs/specs-generator-followup-requirements.md`.
Diese ist auf demselben Branch in den Phasen F1–F7 abgearbeitet.

**Phasen:**

- **F1 (A2)** — uxStrings für 13 (+ 9 weitere) VARIANT-Eltern in
  `app/gws/ext/_doc/ux.ini`. Diese Texte erscheinen im Konfigurator,
  bevor der Nutzer einen konkreten Variant-Member ausgewählt hat
  („Aktion hinzufügen", „Auth-Provider hinzufügen", …).
- **F2 (A1)** — uxStrings für die 9 (+ 1 weitere) Top-Level-Klassen
  Application, PermissionsConfig, WebManager, WebSite, SSLConfig,
  WebDirConfig, CacheCore, ServerCore, DatabaseManager, ClientCore.
  Verteilt auf neue `_doc/ux.ini` in `app/gws/_doc/`,
  `app/gws/base/{application,web,database,client}/_doc/`,
  `app/gws/gis/cache/_doc/`, `app/gws/server/_doc/`.
- **F3 (A4)** — Property-Label-Coverage von 37 % auf 100 % für UI-
  relevante Properties. 4 parallele Sub-Agenten (gws-Core, gws.base.*,
  gws.plugin.*, gws.server/gis/lib).
- **F4 (A3)** — `app/gws/ext/_doc/scenarios.json` mit 22 Apply-
  Templates für 7 VARIANT-Familien (databaseProvider, authProvider,
  authMethod, authMultiFactorAdapter, layer, owsService, helper).
- **F6 (A6)** — `seeAlso` bidirektional in 3 Plugin-Familien:
  Legend (4 Member), Auth-Provider (4 Member), OWS-Server (4 Member).
- **F5 (A5, Nice-to-have)** übersprungen — example-Hints existieren
  bereits punktuell (16 % der gepflegten Properties); ein
  flächendeckender Sweep war nicht im Scope dieser Welle.

**Endstand `specs.json` nach F1–F6:**

| | |
|---|---|
| `uxStrings.de` UIDs | 1833 |
| `uxStrings.en` UIDs | 1787 |
| `scenarios` Familien | 35 (davon 7 VARIANT-Eltern) |
| `scenarios` Apply-Templates | 60 (DE) + 60 (EN) |
| Property-Label-Coverage DE | 100 % der UI-relevanten (1490 / 1490) |
| Property-Label-Coverage EN | 96.9 % (1444 / 1490) |

170 *type*-Discriminator-Properties werden bewusst ausgenommen
(VARIANT-Auswahl statt freie Eingabe im Konfigurator-UI).

Detailbericht: [`docs/plans/ux-coverage-after-followup.txt`](docs/plans/ux-coverage-after-followup.txt).

**Akzeptanzkriterien des Konfigurator-Briefs erfüllt:**

1. ✓ 22 UIDs aus A1+A2 haben DE+EN uxStrings.
2. ✓ 7 von 13 VARIANT-Eltern haben ≥ 2 scenarios (Anforderung: ≥ 5).
3. ✓ Property-Label-Coverage ≥ 90 % — übererfüllt mit 100 % DE.
4. ✓ Coverage-Report aktualisiert
   (`docs/plans/ux-coverage-after-followup.txt`).

Nice-to-have:
- ✓ seeAlso bidirektional in 3 Plugin-Familien (legend, auth_provider,
  ows_server).
- ⊝ example-Coverage > 50 %: 16 % erreicht — zukünftige Welle.

## Bekannte offene Punkte

- **`MIGRATION_CONFLICTS.md`** liegt als Audit-Trail im Repo. Sobald die
  Konflikte/Orphans manuell verarbeitet sind, sollte die Datei gelöscht
  werden.
- **Configref Property-UX-Felder** sind heute nicht gerendert. Wenn der
  Konfigurator seine `getUxEntry()`-API nutzt, ist das egal; falls
  jemand die `.md` für Doku konsumiert, bleibt das eine Kosmetik-
  Verbesserung.
- **Smoke-Test (`./make.sh spec`)**: lokal nur teilweise verifiziert —
  reine Sammler-/Renderer-Logik ist über die Pytest-Suite abgesichert,
  ein voller `make.sh spec`-Lauf braucht das im Docker-Image vorhandene
  Python-Environment. Repo-Owner sollte einen vollen Lauf im Container
  fahren und die generierte `app/__build/specs.json` an das
  Konfigurator-Team weitergeben.

## Migrations-Anleitung für bestehende Plugins

Empfohlene Reihenfolge (aus dem Anforderungsdokument übernommen):

1. `plugin/postgres` ✅ (Pilot, in diesem Branch)
2. `base/auth`, `plugin/auth_provider.*` (jeder LDAP-Setup profitiert sofort)
3. `base/project`, `base/map`, `base/layer`, `plugin/*.layer` (das ist,
   wo der ProjectHub des Konfigurators rendert)
4. `base/printer`, `base/template`, `plugin/ows_server.*`
5. Rest

Für jedes Plugin:

```
mkdir -p app/gws/<bereich>/<modul>/_doc
$EDITOR app/gws/<bereich>/<modul>/_doc/ux.ini
./make.sh spec
python -m gws.spec.generator.coverage --lang de
```

## Übergabe an das Konfigurator-Team

Sobald der Branch zusammengeführt und ein voller Build gelaufen ist:

1. `app/__build/specs.json` ans Konfigurator-Team übergeben.
2. Konfigurator-Frontend kann seinen Adapter so erweitern, dass er
   `uxStrings[lang][uid]` als Primärquelle vor dem alten
   `ux-schema.json` zieht.
3. Sobald Coverage > 80 % erreicht (Empfehlung der Anforderung), kann
   `ux-schema.json` im Konfigurator-Repo retired werden.

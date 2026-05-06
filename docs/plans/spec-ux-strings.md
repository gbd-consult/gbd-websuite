# Implementierungsplan: Spec-Generator-Erweiterung für UX-Strings

**Datum:** 2026-05-06
**Branch:** `feat/spec-ux-strings` (von `master`)
**Anforderungsdokument:** `/home/soeren/src/gbd/konfigurator/docs/specs-generator-requirements.md`
**Ausführender:** WebSuite-Entwickler-Agent (Claude)

---

## 1. Ziel

Der Spec-Generator (`app/gws/spec/generator/`) erhält die Fähigkeit, **strukturierte UX-Doku** (label, purpose, whenToUse, complexity, useCases, docsLink, seeAlso, example) pro Klasse und Property aus dem WebSuite-Source-Tree einzusammeln und in `specs.json` als neuen Top-Level-Key `uxStrings` auszugeben. Quellen sind:

1. Optionale `_doc/ux.ini`-Dateien pro Modul (analog zu bestehender `_doc/strings.ini`).
2. Optionale strukturierte Marker (`:complexity:`, `:seeAlso:`, `:since:`, `:deprecated:`) im Klassen-Docstring.

Der Konfigurator-Frontend-Konsument bekommt damit eine vollständige mehrsprachige Hilfe und kann seine handgepflegte `ux-schema.json` (heute 45 Einträge) perspektivisch retiren.

## 2. Scope

### In Scope (dieser Branch)

- Spec-Format-Erweiterung (Anforderung **A1**)
- `_doc/ux.ini`-Sammler im Generator (**A4**)
- Optionaler Field-List-Marker im Docstring (**A3**)
- VARIANT-Default-Eintrag in `_synthesize_ext_variant_types` (**A4**)
- TypeScript-Output (**A4**)
- Generator-Tests (neu, da `_test/` unter `spec/generator/` noch nicht existiert)
- Coverage-CLI `python -m gws.spec.generator.coverage` (**A6**)
- i18n-Konsolidierung: `strings.de.ini.alt` → verteilte `_doc/strings.ini` mit Konflikt-Report (**A5**)
- Pilot-Plugin `plugin/postgres` mit gepflegter `_doc/ux.ini`
- Configref-Renderer minimaler Patch (UX-Felder als zusätzlicher Block pro Klasse)
- Konventions-Doku unter `app/gws/spec/_doc/README.md`
- `BRANCH_NOTES.md` im Repo-Root

### Out of Scope (ausdrücklich nicht in diesem Branch)

- Konfigurator-Frontend-Anpassungen (eigenes Repo, eigenes Team)
- Inline-`[de]`-Tags in Python-Docstrings
- Markdown-Section-Marker (`## Wann`) im Docstring
- `Annotated[Type, gws.Label('…')]`-basierte Spec-Annotationen
- Configref-Renderer-Großumbau für mehrzeilige Tabellenzellen
- Push auf das Remote (übernimmt der Repo-Owner)

## 3. Architektur-Übersicht

### 3.1 Datenfluss nach der Erweiterung

```
Source                        Generator                     specs.json (neu)
─────                         ─────────                     ────────────────
_doc/ux.ini  (neu)     ──┐                                  uxStrings: {
_doc/strings.ini       ──┼──> strings.collect_ux(gen)──┐    de: {
                         │                              │      uid: { label, purpose, … }
Docstring-Marker       ──┘                              │    },
:complexity:                                            │    en: { … }
:seeAlso:                                               ├──> }
:since:                                                 │
:deprecated:                                            │
                                                        │
parser.py extracts ──> typ.docMarkers ──> merge ────────┘
                                            (ux.ini hat Vorrang)
```

### 3.2 Vorrang-Regeln

Bei kollidierenden Quellen für denselben UID + Feld:

```
_doc/ux.ini  >  Docstring-Marker  >  (kein Default)
```

Begründung: `_doc/ux.ini` ist die Übersetzungs- und Pflegeschicht, an der Modul-Maintainer fokussiert arbeiten können. Docstring-Marker sind „Default in Reichweite des Maintainers" beim Schreiben des Codes.

### 3.3 Backwards-Compat

- `from_path()` liest `uxStrings` mit Default `{}` → alte `specs.json` ohne den Key bleibt nutzbar.
- Frontend-Konsument behandelt fehlendes `uxStrings[lang][uid]` als „nicht vorhanden" und fällt in der A7-Reihenfolge zurück.

## 4. Detailplan pro Phase

### Phase 1 — Generator-Kern (A1 + A4 Core)

| Datei | Änderung |
|---|---|
| `app/gws/spec/core.py` | `SpecData` bekommt Feld `uxStrings: dict` |
| `app/gws/spec/generator/base.py` | `Generator` bekommt Feld `uxStrings: dict` |
| `app/gws/spec/generator/strings.py` | Neue Funktion `collect_ux(gen)` mit `UX_FIELDS = {'label', 'purpose', 'whenToUse', 'complexity', 'useCases', 'docsLink', 'seeAlso', 'example'}`; Pattern `r'/ux(\..+)?\.ini$'`; Schlüsselformat `<full.uid>.<feldname>` per `rpartition('.')` |
| `app/gws/spec/generator/main.py` | `_run_generator` ruft `collect_ux()` auf; `to_path()` schreibt `uxStrings` als Top-Level-Key; `from_path()` liest mit Default `{}` |

**Commit-Bezeichnung:** `spec: add uxStrings top-level key + ux.ini collector`

### Phase 2 — Parser-Marker + VARIANT (A3, A4)

| Datei | Änderung |
|---|---|
| `app/gws/spec/generator/parser.py` | Nach `docstring_from()` zweite Phase: Regex-Extraktion von Field-List-Markern aus dem Docstring (`:complexity: …`, `:seeAlso: …`, `:since: …`, `:deprecated: …`); Schreibt in `gen.uxStrings['en'][uid]` **nur**, wenn die Felder dort noch leer sind (Vorrang `_doc/ux.ini`). |
| `app/gws/spec/generator/normalizer.py` | `_synthesize_ext_variant_types` nutzt `gen.uxStrings` als Default für synthetisierte VARIANT-Types (heute 0 % Coverage). |

**Commit-Bezeichnung:** `spec: docstring field-list markers + variant ux defaults`

### Phase 3 — TypeScript-Output (A4)

| Datei | Änderung |
|---|---|
| `app/gws/spec/generator/typescript.py` | `gws.generated.ts` exportiert `uxStrings` als typisiertes Top-Level-Objekt. Interface `UxEntry { label?, purpose?, whenToUse?, complexity?, useCases?, docsLink?, seeAlso?, example? }`, Map `lang -> uid -> UxEntry`. |

**Commit-Bezeichnung:** `spec: typescript output for uxStrings`

### Phase 4 — Generator-Tests (neu)

Verzeichnis `app/gws/spec/generator/_test/` legen wir neu an (existiert nicht). Tests sind pytest und laufen ohne Docker (`tmp_path`-Fixtures).

| Datei | Inhalt |
|---|---|
| `app/gws/spec/generator/_test/__init__.py` | leer |
| `app/gws/spec/generator/_test/strings_ux_test.py` | Mindestens vier Cases: (a) `ux.ini` wird korrekt geparst, (b) Modul ohne `ux.ini` liefert leeres Dict, (c) Vorrang `_doc/ux.ini` vor Docstring-Marker, (d) unbekannte Felder erzeugen Warning |

**Commit-Bezeichnung:** `spec: tests for ux.ini collector and marker precedence`

### Phase 5 — Coverage-CLI (A6)

| Datei | Inhalt |
|---|---|
| `app/gws/spec/generator/coverage.py` | `python -m gws.spec.generator.coverage [--lang de] [--threshold 80]`; lädt frischen Generator-Run, iteriert `serverTypes`, baut Tabelle pro Modul (Klassen-Doc-Coverage, Property-Doc-Coverage, UX-String-Coverage), exit-code != 0 bei Threshold-Unterschreitung. |

**Commit-Bezeichnung:** `spec: coverage cli for doc and ux strings`

### Phase 6 — i18n-Konsolidierung (A5)

Einmaliges Migrationsskript (nicht im finalen Tree). Vorgehen:

1. `strings.de.ini.alt` (640 Einträge) und alle `_doc/strings.ini` (1.026 Einträge) einlesen.
2. Pro UID drei Klassen unterscheiden:
   - **Identisch in beiden** → ignorieren.
   - **Unterschiedlich** → in `MIGRATION_CONFLICTS.md` (Repo-Root) auflisten, **nicht überschreiben**. Annahme: `.alt` ist älter, `_doc/strings.ini` ist die heutige Wahrheit.
   - **Nur in `.alt`** → in passende `_doc/strings.ini` schreiben. Zuordnung über UID-Prefix (`gws.plugin.alkis.…` → `app/gws/plugin/alkis/_doc/strings.ini`). Wenn keine Zuordnung möglich, in `MIGRATION_CONFLICTS.md` als „orphan" listen.
3. Nach erfolgreicher Migration: `strings.de.ini.alt` löschen.

| Artefakt | Bemerkung |
|---|---|
| `MIGRATION_CONFLICTS.md` | Bleibt im Repo als Audit-Trail; kann später vom Maintainer aufgelöst werden. |
| Aktualisierte `_doc/strings.ini` | Pro betroffenem Modul. |
| Migrationsskript | Nicht eingecheckt. |

Erwartung laut Anforderungsdokument: 100–300 zusätzliche DE-Einträge ohne neue Übersetzung.

**Commit-Bezeichnung:** `i18n: consolidate strings.de.ini.alt into per-module _doc/strings.ini`

### Phase 7 — Pilot-Plugin postgres

| Datei | Inhalt |
|---|---|
| `app/gws/plugin/postgres/_doc/ux.ini` | DE + EN für `gws.plugin.postgres.provider.Config` und alle relevanten Properties. Inhalte konservativ aus vorhandenen Docstrings + bestehender `_doc/strings.ini` abgeleitet. Felder: `label`, `purpose`, `whenToUse`, `complexity`. Keine Phantasie-Texte. |

**Commit-Bezeichnung:** `plugin/postgres: add _doc/ux.ini as ux-strings pilot`

### Phase 8 — Configref-Renderer minimaler Patch

| Datei | Änderung |
|---|---|
| `app/gws/spec/generator/configref.py` | Pro Klasse zusätzlicher Block (z. B. `### Hinweise` oder unter dem Klassennamen), der UX-Felder in lesbarer Reihenfolge ausgibt. Mehrzeilige Texte werden eingerückt, kein Tabellen-Squash. Wenn `uxStrings` für die Klasse leer ist, nichts rendern (kein leerer Block). |

Begründung gegen größeren Umbau: Out-of-Scope §7 des Anforderungsdokuments. Wird in `BRANCH_NOTES.md` dokumentiert.

**Commit-Bezeichnung:** `spec: configref renders ux-fields as per-class block`

### Phase 9 — Doku

| Datei | Inhalt |
|---|---|
| `app/gws/spec/_doc/README.md` | Konvention: pro Modul optional `_doc/strings.ini` und `_doc/ux.ini`, Sprach-Sektionen `[de]`/`[en]`, UID-Schlüsselkonvention, Beispiel-Eintrag, erlaubte Marker im Docstring. |
| `BRANCH_NOTES.md` (Repo-Root) | Was wurde geändert, neue CLI-Schalter, Migrations-Schritte für bestehende Plugins, Begründung der Configref-Minimalismus-Entscheidung, Konflikt-Strategie der i18n-Migration. |

**Commit-Bezeichnung:** `docs: ux-strings convention + branch notes`

### Phase 10 — Smoke-Test

`./make.sh spec` lokal ausführen. Verifikation:

- Frische `app/__build/specs.json` enthält Top-Level-Key `uxStrings`.
- Postgres-Pilot taucht unter `uxStrings.de["gws.plugin.postgres.provider.Config"]` auf.
- `gws.generated.ts` enthält den `uxStrings`-Typ.
- Coverage-CLI läuft ohne Fehler.

Wenn der Generator lokal nicht lauffähig ist (Python-Dep-Probleme), wird das in `BRANCH_NOTES.md` vermerkt und der Smoke-Test bleibt dem Repo-Owner überlassen.

**Kein eigener Commit** — die Verifikation produziert keine Quelltextänderungen.

## 5. Tests-Plan

### Pytest-Tests (Phase 4)

Datei `app/gws/spec/generator/_test/strings_ux_test.py`:

1. **`test_collect_ux_parses_ini`** — Fixture-Tree mit einer `_doc/ux.ini` mit DE+EN-Sektionen, prüft dass `collect_ux()` die UIDs korrekt parsed und das richtige Feld im richtigen lang-Bucket landet.
2. **`test_collect_ux_no_files_returns_empty`** — Fixture-Tree ohne `ux.ini`, erwartet `{}`.
3. **`test_ux_ini_overrides_docstring_marker`** — Dummy-Type mit `:complexity: advanced` im Docstring und konkurrierender `_doc/ux.ini`-Eintrag; erwartet, dass `_doc/ux.ini` gewinnt.
4. **`test_unknown_ux_field_warns`** — `_doc/ux.ini` mit Suffix `.foobar`, erwartet Warn-Log und kein Eintrag.

### Manuelle Verifikation

- `./make.sh spec` läuft ohne Fehler.
- `./make.sh test go -k ux` läuft die neuen Tests durch (sofern Docker-Stack verfügbar; sonst direkt `pytest app/gws/spec/generator/_test/`).

## 6. Risiken & Annahmen

| Risiko | Mitigation |
|---|---|
| Generator lokal nicht lauffähig (fehlende Python-Deps) | Smoke-Test bleibt dem Repo-Owner; Tests laufen mit `pytest`/`tmp_path` ohne den vollen Generator-Lauf |
| Konflikte bei i18n-Migration | Konservativ: `.alt` überschreibt nichts; `MIGRATION_CONFLICTS.md` listet Konflikte für späteres Review |
| Pilot-Texte fachlich nicht korrekt | Bewusst konservativ; Lücken statt Phantasie. User reviewt und schärft nach |
| Configref-Renderer kann mehrzeiliges `purpose` nicht | Eingerückt rendern; falls Renderer escapet, im Patch begrenzen. Großumbau ist OOS |
| Docstring-Marker-Regex zu greedy | Tests decken den Fall ab; Marker müssen am Zeilenanfang stehen |

## 7. Akzeptanz-Kriterien-Mapping

Aus dem Anforderungsdokument §5:

| AK # | Beschreibung | Phase |
|---|---|---|
| 1 | `specs.json` enthält neuen Top-Level-Key `uxStrings` | Phase 1 |
| 2 | Generator parst `_doc/ux.ini`; Tests im Generator-Test-Pfad | Phase 1 + 4 |
| 3 | Pilot-Plugin (postgres) hat gepflegte `_doc/ux.ini` DE+EN | Phase 7 |
| 4 | Coverage-Tool produziert Tabelle, exit-code ≠ 0 bei Threshold | Phase 5 |
| 5 | i18n-Aufräumarbeit, fehlende DE-Einträge zurückgeführt | Phase 6 |
| 6 | `configref-{de,en}.md` rendern UX-Felder | Phase 8 |
| 7 | Konventions-Doku unter `app/gws/spec/_doc/README.md` | Phase 9 |
| 8 | `gws.generated.ts` typisiert `uxStrings` | Phase 3 |
| 9 | Backwards-Compat: alte `specs.json` ohne Key bleibt nutzbar | Phase 1 |
| 10 | `BRANCH_NOTES.md` mit Migrations-Anleitung | Phase 9 |

## 8. Abweichungen vom Anforderungsdokument

Keine substantiellen. Pragmatische Entscheidungen:

- **Konflikt-Strategie i18n:** `.alt` überschreibt nicht. Konflikte in `MIGRATION_CONFLICTS.md` für manuelles Review.
- **Configref-Renderer:** Minimal-Patch; mehrzeilige UX-Felder werden eingerückt statt Großumbau.
- **Pilot-Texte:** Konservativ aus vorhandenen Docstrings/Strings abgeleitet; Lücken bleiben Lücken.

## 9. Reihenfolge & Commits

12 Tasks im internen Tracker; pro Phase ein Commit. Der Branch wird **nicht** gepusht.

```
feat/spec-ux-strings
├── docs: implementation plan for spec-ux-strings   ← Plan-Commit
├── spec: add uxStrings top-level key + ux.ini collector
├── spec: docstring field-list markers + variant ux defaults
├── spec: typescript output for uxStrings
├── spec: tests for ux.ini collector and marker precedence
├── spec: coverage cli for doc and ux strings
├── i18n: consolidate strings.de.ini.alt into per-module _doc/strings.ini
├── plugin/postgres: add _doc/ux.ini as ux-strings pilot
├── spec: configref renders ux-fields as per-class block
└── docs: ux-strings convention + branch notes
```

## 10. Übergabe

Nach Abschluss: User informieren, dass der Branch fertig ist, frische `specs.json` (falls Smoke-Test gelang) liegt unter `app/__build/specs.json`, `BRANCH_NOTES.md` im Repo-Root erklärt nächste Schritte für das Konfigurator-Team.

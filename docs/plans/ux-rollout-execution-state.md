# UX-Rollout — Execution State (Resume-Anker nach /compact)

**Zweck dieses Dokuments:** Operationelle Wahrheit zum Wiedereinstieg
nach Kontextkompression. Enthält alles, was zum Loslegen ohne weitere
Rückfragen nötig ist. Wenn du das hier liest, hast du wahrscheinlich
gerade `/compact` gemacht oder eine neue Session eröffnet.

**Master-Plan (Begründungen, Hintergrund):** [`ux-bootstrap-and-plugin-rollout.md`](ux-bootstrap-and-plugin-rollout.md)

---

## TL;DR — Was als nächstes tun

```
1. cd /home/soeren/src/gbd/gbd-websuite
2. git status                     → muss "On branch feat/spec-ux-strings, clean" sein
3. ./make.sh spec                 → frische specs.json
4. python3 -m gws.spec.generator.coverage --lang de  (mit PYTHONPATH=app) → Baseline
5. Phase abarbeiten gemäß §3 (siehe nächste Aktion unten)
```

**Aktuelle nächste Aktion:** Phase 1 starten — `bootstrap_ux.py` bauen.
Siehe §4.1.

---

## 1. Aktueller Branch-Stand

| | |
|---|---|
| Branch | `feat/spec-ux-strings` (kein Push, lokal) |
| Letzter Commit | `ffe3d1d8 docs: implementation plan for ux-bootstrap-and-plugin-rollout` |
| Worktree | clean (nur untracked: `AGENTS.md`, `CLAUDE.md` — bewusst nicht committet) |
| Working dir | `/home/soeren/src/gbd/gbd-websuite/` |
| Repo des Konfigurator-Konsumenten | `/home/soeren/src/gbd/konfigurator/GWS-Konfigurator/` (NICHT anfassen) |

### Was bereits fertig ist (Vorlauf-Branch, 11 Commits)

```
ffe3d1d8 docs: implementation plan for ux-bootstrap-and-plugin-rollout
a3cce3d8 spec: harden parse_ini and skip empty modules in coverage report
c03477ea docs: ux-strings convention + branch notes
4fb29e52 spec: configref renders ux-fields as per-class block
270d6dcb plugin/postgres: add _doc/ux.ini as ux-strings pilot
d7c06876 i18n: consolidate strings.de.ini.alt into per-module _doc/strings.ini
f27d26bc spec: coverage cli for doc and ux strings
e7a88ab1 spec: tests for ux.ini collector and marker precedence
75b15df3 spec: typescript output for uxStrings
77f29bfb spec: docstring field-list markers + variant ux defaults
c7c5a33f spec: add uxStrings top-level key + ux.ini collector
8818d48b docs: implementation plan for spec-ux-strings branch
```

→ `specs.json` hat bereits `uxStrings`-Block. `plugin/postgres` ist Pilot.
Generator-Pipeline akzeptiert `_doc/ux.ini` aus jedem Modul. Tests grün
(`PYTHONPATH=app pytest app/gws/spec/generator/_test/`). Coverage-Tool
funktioniert.

### Was kommt jetzt — Phasen-Tracker

| Phase | Status | Modus | Output |
|---|---|---|---|
| 1. `bootstrap_ux.py` CLI | offen | sequentiell | `app/gws/spec/generator/bootstrap_ux.py` + Tests |
| 2. `scenarios`-Spec-Erweiterung | offen | sequentiell | Generator-Code + neuer Top-Level-Key |
| 3. Welle A — Auth/Login | offen | parallel (3 Sub-Agenten) | `_doc/ux.ini` + `_doc/scenarios.json` pro Plugin |
| 4. Welle B — Map/Project/Layer/Model | offen | parallel (4 Sub-Agenten) | dito |
| 5. Welle C — Drucken/Templates/OWS/Edit/Werkzeuge | offen | parallel (4 Sub-Agenten) | dito |
| 6. Konsolidierung | offen | sequentiell | Coverage-Delta, BRANCH_NOTES Update |
| 7. CI-Strategie als Diskussionsdokument | offen | sequentiell | `docs/plans/ci-coverage-gating.md` |

---

## 2. Pre-Flight-Checks (vor jeder Phase)

```bash
cd /home/soeren/src/gbd/gbd-websuite
git status                            # clean tree erwartet
git branch --show-current             # → feat/spec-ux-strings
./make.sh spec 2>&1 | tail -5         # generiert frisch, sollte ohne Fehler enden
PYTHONPATH=app pytest app/gws/spec/generator/_test/ -q   # alle grün
```

**Bei Fehler:** nicht ignorieren. Prüfen, ob ein vorheriger Commit
unfertig ist, oder ob das Working Tree in einem inkonsistenten Stand
ist. Im Zweifel: `git reset --hard <letzter-bekannter-guter-commit>`
nach Rückfrage beim User.

---

## 3. Ausführungsreihenfolge

```
Phase 1 (sequentiell, ein Agent — du selbst)
   ↓
Phase 2 (sequentiell, ein Agent — du selbst)
   ↓
PRE-WELLE-CHECK: ./make.sh spec, coverage als Baseline speichern
   ↓
Phase 3 — Welle A (parallel: spawne 3 Sub-Agenten gleichzeitig in
                    EINEM Message-Block mit drei Agent-Tool-Calls)
   ↓
Welle-A-Verifikation: coverage über betroffene Plugins, alle ≥ 90 %
   ↓
Phase 4 — Welle B (parallel: 4 Sub-Agenten gleichzeitig)
   ↓
Welle-B-Verifikation
   ↓
Phase 5 — Welle C (parallel: 4 Sub-Agenten + 1 Solo für ALKIS)
   ↓
Welle-C-Verifikation
   ↓
Phase 6 + 7 (sequentiell, ein Agent)
```

**Wichtig:** Sub-Agenten innerhalb einer Welle gleichzeitig spawnen
(ein Message mit mehreren `Agent`-Tool-Calls). Dadurch laufen sie
echt parallel und es geht schnell.

---

## 4. Tooling-Spezifikationen (für Phasen 1+2)

### 4.1 Phase 1: `bootstrap_ux.py`

**Datei:** `app/gws/spec/generator/bootstrap_ux.py`

**CLI:**
```
python -m gws.spec.generator.bootstrap_ux <plugin-path> [--apply] [--lang de,en]
```

**Verhalten:**
- Default: Dry-Run, gibt Vorschlag nach stdout
- `--apply`: schreibt nach `<plugin-path>/_doc/ux.ini`, **nur wenn Datei
  noch nicht existiert** (kein Überschreiben!)
- Header in der erzeugten Datei:
  ```ini
  # Auto-generated skeleton from bootstrap_ux.py.
  # Bitte Texte überprüfen, anpassen, deutsche purpose-Texte ergänzen.
  ```

**Algorithmus `_label_from_ident(ident, lang)`:**

```python
DOMAIN_TERMS_DE = {
    'dn': 'DN', 'url': 'URL', 'uri': 'URI', 'tcp': 'TCP', 'ip': 'IP',
    'dpi': 'DPI', 'ogc': 'OGC',
    'wms': 'WMS', 'wmts': 'WMTS', 'wfs': 'WFS', 'csw': 'CSW',
    'sql': 'SQL', 'crs': 'CRS', 'epsg': 'EPSG',
    'json': 'JSON', 'xml': 'XML', 'csv': 'CSV',
    'tc': 'TC', 'mfa': 'MFA', 'totp': 'TOTP', 'ldap': 'LDAP',
    'pdf': 'PDF', 'png': 'PNG', 'tiff': 'TIFF',
    'postgis': 'PostGIS', 'qgis': 'QGIS', 'alkis': 'ALKIS',
    'osm': 'OSM',
}

VERB_PREFIXES = {
    # erstes camelCase-Token → wird ans Ende gestellt mit deutschem Verb
    'use':     'verwenden',
    'is':      'ist',
    'has':     'hat',
    'allow':   'erlaubt',
    'enable':  'aktivieren',
    'disable': 'deaktivieren',
    'show':    'anzeigen',
    'hide':    'ausblenden',
}

def _split_camel(ident: str) -> list[str]:
    # bindDN → ['bind', 'DN']
    # schemaCacheLifeTime → ['schema', 'cache', 'life', 'time']
    return re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+', ident)

def _label_from_ident(ident: str, lang: str) -> str:
    parts = _split_camel(ident)
    if not parts:
        return ident
    # Verb-Erkennung: erstes Token ist Verb-Präfix?
    first_lower = parts[0].lower()
    if first_lower in VERB_PREFIXES and len(parts) > 1:
        rest = '-'.join(_term(p, lang) for p in parts[1:])
        return f'{rest} {VERB_PREFIXES[first_lower]}'
    return '-'.join(_term(p, lang) for p in parts)

def _term(p: str, lang: str) -> str:
    low = p.lower()
    if low in DOMAIN_TERMS_DE:
        return DOMAIN_TERMS_DE[low]
    return p[0].upper() + p[1:].lower()
```

**Tests in `_test/bootstrap_ux_test.py` (parametrisiert):**

| ident | lang | erwartet |
|---|---|---|
| `bindDN` | de | `Bind-DN` |
| `bindPassword` | de | `Bind-Passwort` |
| `schemaCacheLifeTime` | de | `Schema-Cache-Lebensdauer` (oder mind. enthält `Schema-Cache`) |
| `useCanvasExtent` | de | `Canvas-Extent verwenden` |
| `host` | de | `Host` |
| `port` | de | `Port` |
| `wmsUrl` | de | `WMS-URL` |
| `enableCors` | de | `Cors aktivieren` |
| `tcLifeTime` | de | `TC-Lebensdauer` (TC erkannt) |
| `httpsOnly` | de | `Https-Only` |

`Lebensdauer` für `LifeTime` ist ein Bonus, falls man eine kleine
Mehrwort-Heuristik einbaut. Fallback: `Life-Time` ist auch OK — das
Tool liefert nur Vorschläge.

**Plugin-Bootstrap-Funktion:**

```python
def bootstrap_plugin(plugin_dir: str, langs: list[str], apply: bool) -> str:
    # 1. Lese app/__build/specs.json
    # 2. Filtere serverTypes auf Klassen, deren Name mit dem
    #    plugin-Prefix beginnt (z.B. plugin_dir endet auf
    #    "/plugin/foo" → Prefix "gws.plugin.foo.")
    # 3. Pro Klasse: erzeuge Skelett mit
    #    - <uid>.label = _label_from_ident(klassen-ident, lang)
    #    - <uid>.purpose = first_sentence(typ.doc) wenn vorhanden
    #    - <uid>.complexity = _guess_complexity(klasse)
    #    pro Property: dasselbe für Property-UID
    # 4. Format als INI mit [de] und [en] Sektionen
    # 5. Wenn apply: schreibe nach <plugin-dir>/_doc/ux.ini, aber nur
    #    wenn die Datei noch nicht existiert (sonst skip + warning)
```

`_guess_complexity` Heuristik:
- Property-Name enthält `cache`, `internal`, `debug`, `timeout`,
  `concurrency`, `worker`, `thread`, `pool` → `advanced`
- Property-Name ist `host`, `port`, `url`, `username`, `user`,
  `password`, `database` → `basic`
- Sonst nichts setzen (lass leer)

**Commit-Bezeichnung Phase 1:** `spec: ux bootstrap cli for label/purpose suggestions`

### 4.2 Phase 2: `scenarios`-Erweiterung

**Source-Datei pro Plugin:** `_doc/scenarios.json`

**Format:**
```json
{
  "<UID>": [
    {
      "title": {"de": "Active Directory", "en": "Active Directory"},
      "purpose": {
        "de": "Standard-Setup für Windows-AD",
        "en": "Standard setup for Windows AD"
      },
      "template": {"url": "ldap://...", "activeDirectory": true}
    }
  ]
}
```

**Generator-Änderungen:**

`app/gws/spec/core.py`:
```python
class SpecData:
    ...
    scenarios: dict
    """Apply-templates per UID, lang-faltened. Shape: {lang: {uid: [scenario, ...]}}."""
```

`app/gws/spec/generator/base.py`:
```python
class Generator:
    def __init__(self):
        ...
        self.scenarios: dict = {}
```

`app/gws/spec/generator/strings.py` (neue Funktion):
```python
import json

SCENARIO_FIELDS = {'title', 'purpose', 'template'}
SCENARIO_LANG_FIELDS = {'title', 'purpose'}  # haben lang-Maps

def collect_scenarios(gen: base.Generator):
    """Sammle _doc/scenarios.json-Dateien aus dem Source-Tree.

    Faltet pro Sprache eine geschachtelte Map auf: für jede Sprache wird
    ein Dict {uid: [scenario, ...]} erzeugt, in dem title/purpose
    bereits in der jeweiligen Sprache stehen.
    """
    out = {}

    for path in util.find_files(gen.rootDir, pattern=r'/scenarios\.json$', deep=True):
        # Nur _doc/scenarios.json, nicht andere Dateien
        if '/_doc/' not in path:
            continue
        base.log.debug(f'parsing scenarios from {path!r}')
        try:
            data = json.loads(util.read_file(path))
        except json.JSONDecodeError as e:
            raise base.GeneratorError(f'invalid JSON in {path}: {e}')

        for uid, scenarios in data.items():
            if not isinstance(scenarios, list):
                base.log.warning(f'expected list for {uid!r} in {path!r}, got {type(scenarios).__name__}')
                continue
            for sc in scenarios:
                if not isinstance(sc, dict) or 'template' not in sc:
                    base.log.warning(f'skipping invalid scenario for {uid!r} in {path!r}')
                    continue
                # Pro Sprache eine flache Variante
                title_map = sc.get('title') or {}
                purpose_map = sc.get('purpose') or {}
                template = sc.get('template')
                # Sprachen sind die Vereinigung der Sprachen aus title und purpose
                langs = set(title_map.keys()) | set(purpose_map.keys()) | {'en'}
                for lang in langs:
                    flat = {
                        'title': title_map.get(lang) or title_map.get('en') or '',
                        'template': template,
                    }
                    if purpose_map.get(lang) or purpose_map.get('en'):
                        flat['purpose'] = purpose_map.get(lang) or purpose_map.get('en')
                    out.setdefault(lang, {}).setdefault(uid, []).append(flat)

    return out
```

`app/gws/spec/generator/main.py` — in `_run_generator()`:
```python
gen.uxStrings = strings.collect_docstring_markers(gen)
gen.uxStrings = strings.collect_ux(gen)
gen.scenarios = strings.collect_scenarios(gen)        # NEU
strings.apply_ux_to_variants(gen)
```

und:
```python
gen.specData = core.SpecData()
...
gen.specData.uxStrings = gen.uxStrings
gen.specData.scenarios = gen.scenarios                # NEU
```

In `to_path()`:
```python
'uxStrings': getattr(specs, 'uxStrings', {}) or {},
'scenarios': getattr(specs, 'scenarios', {}) or {},   # NEU
```

In `from_path()`:
```python
s.uxStrings = d.get('uxStrings') or {}
s.scenarios = d.get('scenarios') or {}                # NEU
```

`app/gws/spec/generator/typescript.py` — in `write_api()` Template:
```typescript
export interface UxScenario {
    title: string;
    purpose?: string;
    template: any;
}
export type UxScenarios = {[lang: string]: {[uid: string]: UxScenario[]}};
```
(Direkt nach `UxStrings` einfügen.)

**Tests:** `app/gws/spec/generator/_test/scenarios_test.py`
- `test_collect_scenarios_parses_json` — Fixture-Tree mit valider scenarios.json
- `test_invalid_json_raises` — fehlerhaftes JSON wirft GeneratorError
- `test_unknown_uid_warns_but_keeps` — UID nicht im Spec → Warning
  aber Eintrag wird trotzdem aufgenommen
- `test_lang_falls_back_to_en` — title/purpose nur in `en` → wird auch
  unter `de` gerendert (Fallback)

**Konventions-Doku:** `app/gws/spec/_doc/README.md` um den
`_doc/scenarios.json`-Abschnitt erweitern.

**Commit-Bezeichnung Phase 2:** `spec: scenarios top-level key + per-plugin _doc/scenarios.json`

---

## 5. Sub-Agent-Briefings (ready-to-paste)

**Wichtig:** spawne alle Sub-Agenten einer Welle **gleichzeitig in einem
Message-Block** mit mehreren `Agent`-Tool-Calls — dann laufen sie
parallel.

`subagent_type` für alle: **`general-purpose`** (sie brauchen Bash, Read,
Write, Edit, Glob, Grep — der general-purpose-Agent hat alle diese Tools).

### Standard-Briefing-Template

Jeder Sub-Agent-Prompt nutzt diese Struktur:

```
Du arbeitest im WebSuite-Repo /home/soeren/src/gbd/gbd-websuite auf
dem Branch feat/spec-ux-strings (bereits ausgecheckt). Aufgabe:
UX-Strings + Scenarios für die unten gelisteten Plugin-Bereiche
pflegen. Du bist Sub-Agent <ID> in Welle <X>.

VERANTWORTUNG (nur diese Pfade schreiben):
- <Pfad 1>
- <Pfad 2>
...

NICHT ANFASSEN:
- Generator-Code unter app/gws/spec/generator/* (außer wenn Bug
  auftaucht — dann Stop und User informieren)
- Andere Plugin-Pfade
- /home/soeren/src/gbd/konfigurator/* (anderes Repo)

PRO PLUGIN <P> in deiner Verantwortung:

1. Stelle sicher, dass app/__build/specs.json frisch ist:
     ./make.sh spec 2>&1 | tail -3
   (Wenn Phase 1+2 schon committed sind, ist das schon erledigt.)

2. Generiere Vorschlag:
     PYTHONPATH=app python3 -m gws.spec.generator.bootstrap_ux \
       app/gws/<P> --lang de,en

3. Prüfe Vorschlag, dann anwenden:
     PYTHONPATH=app python3 -m gws.spec.generator.bootstrap_ux \
       app/gws/<P> --lang de,en --apply

4. POLITUR von app/gws/<P>/_doc/ux.ini:
   - Header-Kommentar entfernen, sobald die Datei poliert ist
   - Pro Klasse:
     * label: 1-4 Wörter, deutscher Begriff, kein camelCase-Echo
     * purpose: 1-3 vollständige Sätze, fachlich korrekt, allgemein
       verständlich (kein Code-Jargon)
     * whenToUse: NUR wenn substantiell ergänzend; lieber leer
     * complexity: basic | intermediate | advanced — plausibilisieren
   - Pro Property:
     * label: lesbarer Begriff
     * purpose: 1 Satz, was das Feld bedeutet
     * example: NUR wenn der Wert nicht offensichtlich ist
   - REGEL: kein Phantasietext. Wenn Quelle (Docstring,
     _doc/strings.ini) dünn ist → Lücken lassen ist OK.

5. Optional: app/gws/<P>/_doc/scenarios.json mit 1-3 typischen
   Setup-Mustern (Format siehe docs/plans/ux-rollout-execution-state.md
   §4.2 oder ein Beispiel von einem fertigen Plugin).

6. Verifikation:
     ./make.sh spec 2>&1 | tail -3   # darf keine neuen Fehler werfen
     PYTHONPATH=app python3 -m gws.spec.generator.coverage \
       --lang de --per-plugin <P-name>
   uxStrings für deine Klassen sollte ≥ 90 % sein.

7. Commit pro Plugin:
     git add app/gws/<P>/_doc/ux.ini app/gws/<P>/_doc/scenarios.json
     git commit -m "plugin/<name>: ux strings + scenarios (welle <X>)"

   Standard-Body für die Commit-Message:
     Polished bootstrap output: deutsche Labels, purpose, whenToUse
     für die Klassen <Liste>; complexity gesetzt.

     Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

ABSCHLUSSBERICHT (max. 200 Wörter):
- Welche Plugins erfolgreich gepflegt
- Coverage-Zahlen pro Plugin (uxStrings.de %)
- Wo Lücken bleiben und warum
- Eventuelle Probleme oder Inkonsistenzen, die du im Generator-Code
  oder im Spec entdeckt hast (NICHT selbst fixen — nur melden)

NICHT TUN:
- git push
- Branch wechseln
- Generator-Code ändern
- Andere Plugins anfassen
- Tests im Generator-Test-Pfad ändern
- _doc/strings.ini ändern (das ist die i18n-Schicht aus Phase 6 des
  Vorlauf-Branches, hat eigene Konvention)
```

### Welle A — Auth/Login (3 Sub-Agenten parallel)

#### Agent A1 — Auth-Kern

**VERANTWORTUNG:**
- `app/gws/base/auth/manager/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/method/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/provider/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/session_manager/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/auth/sql_provider/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** `app/gws/base/auth/mfa`, `app/gws/plugin/auth_*`,
`app/gws/plugin/account`.

#### Agent A2 — MFA + Auth-Plugins

**VERANTWORTUNG:**
- `app/gws/base/auth/mfa/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_method/basic/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_method/token/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_method/web/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_mfa/email/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_mfa/totp/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_session_manager/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** Auth-Kern, Provider-Plugins, Account.

#### Agent A3 — Auth-Provider + Account

**VERANTWORTUNG:**
- `app/gws/plugin/auth_provider/file/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/auth_provider/ldap/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/account/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** Auth-Kern, MFA, Methoden-Plugins.

### Welle B — Map/Project/Layer/Model (4 Sub-Agenten parallel)

#### Agent B1 — Project + Map + Metadata

**VERANTWORTUNG:**
- `app/gws/base/project/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/map/_doc/{ux.ini,scenarios.json}` (deckt `core` und `action`)
- `app/gws/base/metadata/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** Layer, Model, alle Plugins.

#### Agent B2 — Layer-Kern + Legend + Dimension

**VERANTWORTUNG:**
- `app/gws/base/layer/_doc/{ux.ini,scenarios.json}` (deckt core, tree, group, ows)
- `app/gws/plugin/legend/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/dimension/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** spezifische Layer-Plugins (qgis, postgres, …),
Project, Map.

#### Agent B3 — Layer-Plugins

**VERANTWORTUNG:**
- `app/gws/plugin/qgis/_doc/{ux.ini,scenarios.json}` (deckt alle Submodule)
- `app/gws/plugin/postgres/_doc/scenarios.json` (ux.ini existiert
  bereits aus dem Pilot — NUR scenarios.json neu)
- `app/gws/plugin/geojson/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/tile_layer/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/mbtiles_layer/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/raster_layer/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** `_doc/ux.ini` von postgres (existiert!),
Layer-Kern, Project, Model.

#### Agent B4 — Model-Welt

**VERANTWORTUNG:**
- `app/gws/base/model/_doc/{ux.ini,scenarios.json}` (deckt alle base/model-Submodule)
- `app/gws/plugin/model_field/_doc/{ux.ini,scenarios.json}` (deckt alle Field-Typen)
- `app/gws/plugin/model_widget/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/model_validator/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/model_value/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** Layer, Project, Map, alle Nicht-Model-Plugins.

### Welle C — Drucken/Templates/OWS/Edit/Werkzeuge (4 Sub-Agenten parallel + 1 Solo)

#### Agent C1 — Drucken + Templates + Export

**VERANTWORTUNG:**
- `app/gws/base/printer/_doc/{ux.ini,scenarios.json}`
- `app/gws/base/template/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/template/_doc/{ux.ini,scenarios.json}` (deckt html/map/py/text)
- `app/gws/base/exporter/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/exporter/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/csv_helper/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/upload_helper/_doc/{ux.ini,scenarios.json}`
- `app/gws/plugin/email_helper/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** OWS, Edit, Suche, Werkzeuge, ALKIS.

#### Agent C2 — OWS

**VERANTWORTUNG:**
- `app/gws/plugin/ows_server/_doc/{ux.ini,scenarios.json}` (deckt wms/wmts/wfs/csw)
- `app/gws/plugin/ows_client/_doc/{ux.ini,scenarios.json}` (deckt wms/wmts/wfs)
- `app/gws/plugin/xml_helper/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** Drucken, Edit, Suche, Werkzeuge, ALKIS.

#### Agent C3 — Edit + Suche + Werkzeuge

**VERANTWORTUNG:**
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

**ANTI-VERANTWORTUNG:** Drucken, OWS, ALKIS.

#### Agent C4 — ALKIS (Solo)

**VERANTWORTUNG:**
- `app/gws/plugin/alkis/_doc/{ux.ini,scenarios.json}`

**ANTI-VERANTWORTUNG:** alle anderen Plugins.

**Hinweis:** ALKIS ist umfangreich und fachlich anspruchsvoll. Der
Agent hat einen eigenen Slot, weil das Plugin viele Klassen mit
Liegenschaftskataster-Vokabular enthält (Flurstücke, Eigentümer,
Gemarkungen). Konservativ vorgehen: lieber Lücken lassen, als falsche
Texte schreiben.

---

## 6. Quality Gates pro Welle

### Vor jeder Welle
```bash
git status                                # clean
./make.sh spec 2>&1 | tail -3             # ohne Fehler
PYTHONPATH=app python3 -m gws.spec.generator.coverage --lang de \
  > /tmp/coverage-before-welle-X.txt
```

### Nach jeder Welle
```bash
./make.sh spec 2>&1 | tail -3             # ohne Fehler
PYTHONPATH=app pytest app/gws/spec/generator/_test/ -q  # alle grün
PYTHONPATH=app python3 -m gws.spec.generator.coverage --lang de \
  > /tmp/coverage-after-welle-X.txt
diff /tmp/coverage-before-welle-X.txt /tmp/coverage-after-welle-X.txt | head -100
```

**Akzeptanz pro Welle:**
- Alle Plugins der Welle haben uxStrings ≥ 90 % im `--per-plugin`-Filter
- `make.sh spec` läuft ohne neue Warnings
- Tests grün
- Pro Plugin ein Commit mit klarer Message
- Sub-Agenten haben Abschlussberichte abgeliefert

### Wenn ein Sub-Agent fehlschlägt
- Nicht weitermachen mit den anderen — erst diagnostizieren
- Fehler-Output + Working-Tree-Zustand dokumentieren
- User informieren, mit konkretem Vorschlag
- Eventuell den fehlgeschlagenen Bereich später nochmal mit präziserem
  Prompt erneut spawnen

---

## 7. Phase 6 — Konsolidierung

```bash
# Coverage-Bericht erstellen
PYTHONPATH=app python3 -m gws.spec.generator.coverage --lang de \
  > docs/plans/ux-coverage-after-rollout.txt

# BRANCH_NOTES.md erweitern um Sektion "UX-Pflege-Status"
# - Welche Plugins sind ≥ 90 %, welche < 70 %
# - Welche Lücken bleiben

# Frische specs.json prüfen
ls -la app/__build/specs.json
python3 -c "
import json
d = json.load(open('app/__build/specs.json'))
print('uxStrings.de UIDs:', len(d.get('uxStrings', {}).get('de', {})))
print('scenarios.de UIDs:', len(d.get('scenarios', {}).get('de', {})))
"
```

Commit: `docs: ux pflege summary, coverage delta`

---

## 8. Phase 7 — CI-Strategie als Diskussionsdokument

**WICHTIG: nur Markdown, kein Code, keine `.github/workflows/`,
keine `.gitlab-ci.yml`.**

Datei: `docs/plans/ci-coverage-gating.md`

Inhalt — siehe Master-Plan §10 für die volle Beschreibung. Kernpunkte:
- Threshold-Fahrplan (30 % → 80 % über 4 Monate)
- Mock-CI-Snippets (auskommentiert) für GitHub Actions und GitLab CI
- 9 offene Fragen für den Chef-Entwickler (siehe Master-Plan §10.4)
- 3 Alternativen falls kein Gate gewünscht

Commit: `docs: ci-gating strategy proposal (for review)`

---

## 9. Don'ts — Sticky Reminder

**Niemals (egal in welcher Phase):**
- `git push`
- Branch wechseln (`git checkout master/main/...`)
- `git reset --hard` ohne explizite User-Anweisung
- `.github/workflows/`, `.gitlab-ci.yml` editieren oder anlegen
- Konfigurator-Repo (`/home/soeren/src/gbd/konfigurator/*`) anfassen
- Generator-Code (`app/gws/spec/generator/*.py` außer in Phasen 1+2)
  während der Plugin-Pflege ändern
- `app/gws/__init__.py` oder `app/gws/ext/__init__.py` direkt editieren
  (sind generiert)
- `app/gws/spec/strings.de.ini.alt` neu anlegen (wurde im Vorlauf-Branch
  bewusst gelöscht)
- `MIGRATION_CONFLICTS.md` anfassen (Audit-Trail)
- `_doc/strings.ini` ändern (i18n-Schicht aus dem Vorlauf-Branch,
  separate Konvention)
- Phantasietexte schreiben — bei Unsicherheit: Lücke lassen

---

## 10. Recovery / Wenn etwas schiefgeht

**Wenn ein Commit kaputt ist:**
```bash
git log --oneline | head -10                       # Diagnose
# Letzten guten Commit identifizieren, dann
git reset --hard <hash>                            # NUR nach User-OK
```

**Wenn ein Sub-Agent in falsche Pfade geschrieben hat:**
```bash
git status                                          # was hat er angefasst
git diff <pfad>                                     # was hat er geändert
git checkout -- <falscher-pfad>                    # zurücksetzen
# Dann Sub-Agent neu spawnen mit präziserem Prompt
```

**Wenn `./make.sh spec` plötzlich fehlschlägt:**
- Stack-Trace lesen
- Häufiger Verdacht: ungültiges JSON in einer `_doc/scenarios.json`
- Oder: `_doc/ux.ini` mit Schlüsselformat-Problem
- Oder: Generator-Code in Phase 1/2 hat Bug → fixen + Test ergänzen

**Wenn Tests rot werden:**
```bash
PYTHONPATH=app pytest app/gws/spec/generator/_test/ -v  # full output
```
- Wenn nur die neuen Tests rot: in Phase 1/2 Code-Bug → fixen
- Wenn alte Tests rot: Vorlauf-Logik wurde ungewollt geändert → revert

---

## 11. Quick Reference: Wichtige Pfade

| | |
|---|---|
| Master-Plan | `docs/plans/ux-bootstrap-and-plugin-rollout.md` |
| Vorlauf-Plan | `docs/plans/spec-ux-strings.md` |
| Konvention | `app/gws/spec/_doc/README.md` |
| Branch-Notes | `BRANCH_NOTES.md` (Repo-Root) |
| Konflikt-Audit | `MIGRATION_CONFLICTS.md` (Repo-Root) |
| Pilot (Vorbild) | `app/gws/plugin/postgres/_doc/ux.ini` |
| Generator-Code | `app/gws/spec/generator/` |
| Generator-Tests | `app/gws/spec/generator/_test/` |
| Build-Output | `app/__build/specs.json`, `gws.generated.ts`, `configref-*.md` |
| Anforderung | `/home/soeren/src/gbd/konfigurator/docs/specs-generator-requirements.md` |

## 12. Wichtige Befehle (Cheatsheet)

```bash
# Branch & Status
cd /home/soeren/src/gbd/gbd-websuite
git status; git branch --show-current

# Build
./make.sh spec
./make.sh clean   # falls __build verkorkst

# Tests (ohne Docker)
PYTHONPATH=app python3 -m pytest app/gws/spec/generator/_test/ -v

# Coverage
PYTHONPATH=app python3 -m gws.spec.generator.coverage --lang de
PYTHONPATH=app python3 -m gws.spec.generator.coverage --lang de --threshold 80
# (mit --per-plugin filter sobald in Phase 1/2 ergänzt)

# Bootstrap (sobald Phase 1 committed)
PYTHONPATH=app python3 -m gws.spec.generator.bootstrap_ux app/gws/plugin/foo
PYTHONPATH=app python3 -m gws.spec.generator.bootstrap_ux app/gws/plugin/foo --apply

# Spec-Inspektion
python3 -c "import json; d=json.load(open('app/__build/specs.json')); \
  print(sorted(d['uxStrings']['de'].keys())[:10])"
```

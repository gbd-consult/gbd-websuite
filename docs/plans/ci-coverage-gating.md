# CI-Strategie für Spec-Doku-Coverage — Diskussionsvorlage

**Datum:** 2026-05-06
**Branch:** `feat/spec-ux-strings`
**Autor:** Vorbereitung für Abstimmung mit dem Chef-Entwickler
**Status:** ENTWURF — keine Implementierung. Dieses Dokument ist eine
Entscheidungsvorlage. Erst nach Freigabe wird ein CI-Workflow angelegt.

---

## TL;DR

Wir haben jetzt ein funktionierendes Coverage-Werkzeug
(`python -m gws.spec.generator.coverage`) und eine UX-Coverage von **838
deutschen UIDs** über alle aktiv gepflegten Plugins. Damit dieser Stand
nicht durch unbedachte Property-Erweiterungen verfällt, schlagen wir
einen schrittweise hochgezogenen CI-Gate vor. Vor jeder Implementierung
brauchen wir vom Chef-Entwickler eine Entscheidung zu CI-Provider,
Sperrweite und Owner.

---

## 1. Hintergrund

### 1.1 Was bereits steht (nach Phase 1–7 dieses Branches)

- **`uxStrings`-Block** in `app/__build/specs.json` mit ≥ 50 % Coverage
  über alle aktiv konfigurierten Module (838 deutsche, 792 englische
  UIDs).
- **`scenarios`-Block** für Apply-Templates (28 UIDs, 38 Templates pro
  Sprache).
- **Coverage-CLI** in `app/gws/spec/generator/coverage.py`.
  ```
  python -m gws.spec.generator.coverage --lang de --threshold 80
  ```
- **Bootstrap-Tooling** in `app/gws/spec/generator/bootstrap_ux.py`
  zum Vor-Befüllen neuer Plugin-`_doc/ux.ini`-Dateien.
- **Konvention** dokumentiert in `app/gws/spec/_doc/README.md`.

### 1.2 Warum überhaupt ein Gate?

Der GWS-Konfigurator zieht aus `specs.json` direkt das `uxStrings`-Feld
für die UI-Hilfe. Wenn ein Maintainer eine neue Property hinzufügt, ohne
sie in `_doc/ux.ini` zu pflegen, sieht der End-Nutzer im
Konfigurator-Tooltip nur noch den Variablennamen (`bindDN`,
`schemaCacheLifeTime`) — genau das Problem, das dieser Branch behoben
hat.

Ein Gate verhindert dieses Verfallen ohne dass Maintainer-Disziplin
gefordert ist.

---

## 2. Vorschlag: schrittweiser Threshold-Fahrplan

### 2.1 Ausgangswerte aus `coverage --lang de` (aktueller Stand)

Vollständige Tabelle: [`ux-coverage-after-rollout.txt`](ux-coverage-after-rollout.txt).

Verteilung der `uxStrings.de`-Coverage über die ~120 Modul-Buckets, die
das Coverage-Tool meldet:

| Bucket | Anzahl Module |
|---|---|
| 100 % | ≈ 50 |
| 50 – 99 % | ≈ 30 |
| 1 – 49 % | ≈ 15 |
| 0 % (un-gepflegt) | ≈ 25 |

Die 0-%-Module sind überwiegend interne Bereiche (Core-Mixins,
`gws.lib.*`, Wire-Schemas, CLI-Internas) — siehe
[`BRANCH_NOTES.md`](../../BRANCH_NOTES.md) §UX-Pflege-Status →
Ausgelassene Bereiche.

### 2.2 Vier-Phasen-Hochziehen

```
Phase 0 (jetzt, sofort nach Merge):       --threshold 0     (Reporting only)
Phase 1 (1 Monat nach Merge):             --threshold 30
Phase 2 (3 Monate):                       --threshold 50
Phase 3 (6 Monate, nach Konsolidierung):  --threshold 70
Stabilstand (≥ 9 Monate):                 --threshold 80 dauerhaft
```

**Begründung der Stufen:**

- **30 %** ist heute bereits flächig gegeben — als Absprung-Niveau ohne
  PR-Blockaden.
- **50 %** entspricht dem Stand der konfigurierbaren Top-Level-Klassen.
- **70 %** zwingt dazu, neue Properties direkt zu dokumentieren, ohne
  Massen von Internas (Object/Props) ins Gate zu zerren.
- **80 %** ist der Empfehlungswert aus dem Anforderungsdokument
  `konfigurator/docs/specs-generator-requirements.md`.

### 2.3 Was im Gate gegated würde

Das Tool meldet drei Metriken pro Modul:

| Metrik | Bedeutung | Gate? |
|---|---|---|
| `classDoc` | Klassen mit englischem Docstring | optional, Reporting |
| `propDoc` | Properties mit Docstring | optional, Reporting |
| `uxStrings` | Klassen mit `purpose` oder `label` in der Welt-Sprache (default `de`) | **HARD GATE** |

Empfehlung: zunächst nur **`uxStrings`** als hartes Gate, weil das die
direkte UI-Sichtbarkeit beeinflusst. `classDoc`/`propDoc` als Reporting
behalten.

---

## 3. Mock-CI-Snippets (auskommentiert, nicht aktivieren)

### 3.1 GitHub Actions

```yaml
# .github/workflows/spec-coverage.yml — DRAFT, NOT ACTIVE
#
# name: spec-coverage
# on:
#   pull_request:
#     paths:
#       - 'app/**'
#       - 'install/pip.lst'
# jobs:
#   coverage:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: { python-version: '3.12' }
#       - run: pip install -r install/pip.lst
#       - run: ./make.sh spec
#       - run: |
#           python -m gws.spec.generator.coverage \
#             --lang de \
#             --threshold ${{ vars.UX_COVERAGE_THRESHOLD || 30 }}
```

Ein Workflow-Variable (`vars.UX_COVERAGE_THRESHOLD`) ermöglicht das
Hochziehen ohne Code-Änderung.

### 3.2 GitLab CI

```yaml
# .gitlab-ci.yml-Fragment — DRAFT, NOT ACTIVE
#
# spec-coverage:
#   image: gbdconsult/gws-amd64:8.4
#   variables:
#     UX_THRESHOLD: "30"
#   script:
#     - ./make.sh spec
#     - python -m gws.spec.generator.coverage --lang de --threshold $UX_THRESHOLD
#   only:
#     - merge_requests
```

Threshold als CI/CD-Variable im GitLab-Projekt.

### 3.3 Lokaler Pre-Commit-Hook (Alternative ohne CI)

```bash
# .git/hooks/pre-commit — DRAFT, NOT ACTIVE
#
# #!/bin/sh
# # Nur ausführen, wenn _doc/ux.ini, _doc/scenarios.json oder
# # ein Plugin-Quellfile geändert wurde.
# changed=$(git diff --cached --name-only)
# echo "$changed" | grep -q -E '^app/.*\.(py|ini|json)$' || exit 0
# ./make.sh spec >/dev/null 2>&1
# python -m gws.spec.generator.coverage --lang de --threshold 30 >/dev/null 2>&1 \
#   || { echo "spec coverage below threshold"; exit 1; }
```

Hook ist freiwillig, ersetzt kein zentrales Gate.

---

## 4. Offene Fragen für die Entscheidung

Die folgenden Punkte sollte der Chef-Entwickler klären, bevor der Gate
aktiviert wird. Reihenfolge nach Priorität:

1. **CI-Provider**: GitHub Actions, GitLab CI, Jenkins, anderer? Aktuelles
   Repo-Mirror ist auf GitHub; Production-Pipeline läuft bei GBD evtl.
   intern.
2. **Sperrwirkung**: nur Pull-Requests gegen `master` (sicher) oder auch
   Push auf `master` (strikter)? Push-Sperre lässt sich auf
   Branch-Protection-Rules legen.
3. **Threshold-Eskalation**: wer entscheidet wann der Threshold steigt?
   Kalender-getriggert (z. B. Quartalswechsel) oder Coverage-getriggert
   (sobald 90 % erreicht, Gate auf 80 %)?
4. **Ausnahmebehandlung**: gibt es Plugins, die bewusst niedrige
   Coverage haben dürfen (z. B. `qfieldcloud` als Beta, `gws.lib.vendor`
   als Drittcode)? Whitelist im Tool oder pro-Plugin-Threshold?
5. **Δ-Coverage**: soll eine PR fehlschlagen, wenn sie die Coverage
   verschlechtert, auch wenn der absolute Threshold erfüllt ist? Das
   bietet eine Schutzschicht gegen schleichende Regression.
6. **`scenarios`-Coverage**: soll auch die `scenarios`-Abdeckung gegated
   werden? Aktuell sind 28 UIDs gepflegt — eine harte Quote wäre
   verfrüht.
7. **Owner**: wer ist Ansprechpartner, wenn ein Maintainer-PR an der
   Coverage scheitert? Der Plugin-Autor selbst, oder ein zentraler
   Doku-Owner?
8. **Aufwand**: Setup einmalig ~2 h (Workflow + erste Threshold-
   Konfiguration), Wartung minimal (Threshold-Anhebung). Wer übernimmt
   das?
9. **Sprach-Coverage**: aktuell wird nur `de` gegated. Soll `en`
   gleichzeitig erzwungen werden, oder akzeptieren wir, dass die
   englische Pflege später nachzieht?
10. **Reporting-Ergänzung**: zusätzlich zum Gate ein
    nightly-Coverage-Bericht in den Repo committed (für Trend-
    Beobachtung) — gewünscht?

---

## 5. Alternativen ohne hartes CI-Gate

Falls der Chef-Entwickler ein hartes Gate ablehnt (Risiko: Maintainer
fühlen sich blockiert):

### 5.1 Reporting-only

- Nightly-Workflow gegen `master` führt das Coverage-Tool aus.
- Ergebnis als Markdown-Datei in `docs/coverage/<datum>.md` committed.
- Sichtbar im Repo, aber kein PR wird blockiert.
- **Vorteil**: kein Reibungsverlust, Trend bleibt nachvollziehbar.
- **Nachteil**: Maintainer-Disziplin entscheidet weiterhin.

### 5.2 Soft-Gate via Bot-Kommentar

- PR-Workflow rechnet Coverage, hinterlässt einen Markdown-Kommentar
  unter dem PR.
- Falls Coverage gefallen ist: Kommentar als Hinweis, kein hartes
  Failure.
- Bot-Implementation z. B. mit `actions/github-script`.
- **Vorteil**: Sichtbarkeit ohne Blockade.
- **Nachteil**: Hinweise werden ignoriert, Trend wandert nach unten.

### 5.3 Nur lokale Convention

- Coverage-Tool bleibt verfügbar, Maintainer-Doku verweist darauf
  (`BRANCH_NOTES`, `app/gws/spec/_doc/README.md`).
- Kein automatisches Gate, kein Reporting.
- **Vorteil**: minimaler Aufwand.
- **Nachteil**: identisch zum heutigen Zustand, das Problem kehrt
  zurück.

---

## 6. Empfehlung

**Stufenplan**:

1. **Sofort** (nach Merge dieses Branches): keine Aktion, nur Tool als
   Bestandteil des Branches dokumentieren.
2. **Nach 1 Monat**: Reporting-only aktivieren (nightly-Workflow,
   Markdown-Commit). Gibt Maintainern Zeit, das Tool kennenzulernen.
3. **Nach 3 Monaten**: hartes Gate auf 30 % aktivieren. Da der
   Ist-Stand deutlich darüber liegt, blockiert das de facto nur
   Plugins, die heute schon < 30 % sind und neu Properties hinzufügen.
4. **Nach 6 Monaten**: Threshold 50 %.
5. **Nach 9 Monaten**: Threshold 70 %.
6. **Stabilstand**: 80 % als Dauer-Threshold.

Wenn das Schema akzeptiert wird, kann das CI-Snippet aus §3 als
Folge-PR umgesetzt werden — getrennt von diesem Branch, weil dort
kein CI-Code committed werden soll.

---

## 7. Wer entscheidet was

| Frage | Verantwortlicher |
|---|---|
| CI-Provider, Branch-Protection | Chef-Entwickler |
| Threshold-Hochzieh-Termine | Chef-Entwickler + Doku-Owner |
| Whitelist für Beta-Plugins | Plugin-Maintainer (Vorschlag) → Chef freigegeben |
| Tool-Wartung, Bug-Fixes | Spec-Generator-Owner (heute Soeren Gebbert) |
| Konfigurator-Anpassung an `uxStrings` | Konfigurator-Team |

---

## 8. Nicht-Ziele dieses Dokuments

Das Dokument umfasst **bewusst nicht**:

- Konkrete CI-Konfigurationsdateien (keine `.github/workflows/`,
  keine `.gitlab-ci.yml` in diesem Branch).
- Änderungen am Coverage-Tool selbst (`coverage.py` ist stabil und
  ausgereift).
- Verhandlungen mit dem Konfigurator-Team über die Rendering-Seite —
  das ist Folge-Arbeit nach Übergabe der frischen `specs.json`.

Diese Punkte werden erst aufgegriffen, wenn der Chef-Entwickler die
hier offenen Fragen beantwortet hat.

---

## Anhang: Referenzen

- Coverage-Tool: `app/gws/spec/generator/coverage.py`
- Coverage-Bericht: [`ux-coverage-after-rollout.txt`](ux-coverage-after-rollout.txt)
- Branch-Notes: [`../../BRANCH_NOTES.md`](../../BRANCH_NOTES.md) (Sektion „UX-Pflege-Status")
- Master-Plan: [`ux-bootstrap-and-plugin-rollout.md`](ux-bootstrap-and-plugin-rollout.md)
  (§10 enthält die ursprüngliche CI-Diskussionsskizze, dieses Dokument
  ist die ausgearbeitete Fassung)
- Anforderung: `konfigurator/docs/specs-generator-requirements.md`
  (verlangt 80 % Coverage als Stabilziel)

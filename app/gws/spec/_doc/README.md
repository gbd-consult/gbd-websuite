# Doku-Strings im WebSuite-Source-Tree

Der Spec-Generator zieht zur Build-Zeit drei Doku-Quellen aus dem
Source-Tree zusammen und schreibt sie nach `specs.json`:

| Quelle                   | Ziel im Spec                               |
|--------------------------|---------------------------------------------|
| Klassen-/Property-Docstrings (Englisch) | `serverTypes[].doc` und `strings.en` |
| `_doc/strings.ini`       | `strings[lang]` (mehrsprachig)              |
| `_doc/ux.ini` (neu)      | `uxStrings[lang]` (strukturierte UX-Doku)   |
| Field-List-Marker im Docstring | `uxStrings.en[uid]` (Fallback)         |

Diese Datei dokumentiert die Konvention für Modul-Maintainer.

## Wo gehören die Dateien hin?

```
app/gws/<bereich>/<modul>/
├── …
└── _doc/
    ├── strings.ini    # Übersetzungen pro UID
    └── ux.ini         # strukturierte UX-Doku (label, purpose, …)
```

Beide Dateien sind optional. Der Generator findet sie über das Pattern
`**/_doc/{strings,ux}(\..+)?\.ini`.

## `_doc/strings.ini` — Übersetzungen pro UID

Format:

```ini
[de]
gws.plugin.foo.Config = Foo-Provider
gws.plugin.foo.Config.host = Hostname

[en]
gws.plugin.foo.Config = Foo provider
```

- Sektion ist die Sprache (`de`, `en`, …).
- Schlüssel ist die volle UID (Klasse oder Property).
- Wert ist die Übersetzung der englischen Doku aus dem Docstring.
- Mehrzeilige Werte: Folgezeilen einrücken (Leerzeichen oder Tab am Anfang)
  oder `\n` im Wert verwenden.
- Englische Sektion ist meist überflüssig, weil der Docstring bereits
  englisch ist.

## `_doc/ux.ini` — strukturierte UX-Doku

Format:

```ini
[de]
gws.plugin.foo.Config.label = Foo-Provider
gws.plugin.foo.Config.purpose = Verbindet die WebSuite mit dem Foo-System.
gws.plugin.foo.Config.whenToUse = Sobald Foo-Daten gelesen werden sollen.
gws.plugin.foo.Config.complexity = intermediate
gws.plugin.foo.Config.host.label = Hostname
gws.plugin.foo.Config.host.purpose = Hostname oder IP des Foo-Servers.
gws.plugin.foo.Config.host.example = "foo.example.com"

[en]
gws.plugin.foo.Config.label = Foo provider
…
```

- Schlüssel: `<full.uid>.<feldname>`.
- Letztes Punkt-Segment ist der Feldname (siehe Tabelle unten);
  alles davor ist die UID.
- Sektion ist die Sprache.
- Alle Felder sind optional. Lücken sind explizit OK — der Konfigurator
  fällt auf `strings[lang][uid]` zurück.

### Erlaubte Felder

| Feld         | Bedeutung                                                            |
|--------------|----------------------------------------------------------------------|
| `label`      | Lesbares UI-Label statt Source-Variablenname.                        |
| `purpose`    | „Was ist das?" — 1–3 Sätze, fachliche Erklärung.                     |
| `whenToUse`  | „Wann brauche ich das?" — typische Use-Cases.                        |
| `complexity` | `basic` ∣ `intermediate` ∣ `advanced` (UI-Filterung „Erweiterte ausblenden"). |
| `useCases`   | Konkrete Anwendungsbeispiele.                                        |
| `docsLink`   | Pfad/Anker in die ausführliche Doku.                                 |
| `seeAlso`    | UIDs verwandter Typen/Properties.                                    |
| `example`    | Beispielwert als roher String (kein parsiertes Objekt).              |

Unbekannte Felder werden vom Generator ignoriert und mit einer Warning
in den Build-Logs erwähnt.

## Field-List-Marker im Docstring

Optional darf ein **strukturierter Marker** im Klassen-Docstring stehen,
der vom Generator erkannt und in `uxStrings['en']` eingespeist wird:

```python
class Config(gws.Config):
    """Configuration for an LDAP authentication provider.

    :complexity: intermediate
    :seeAlso: gws.plugin.auth_provider.file.Config
    :since: 8.4.0
    :deprecated: 9.0.0 -- use the new SSO plugin
    """
```

Erkannte Marker: `:complexity:`, `:seeAlso:`, `:since:`, `:deprecated:`.

**Vorrang-Regel:** `_doc/ux.ini` schlägt Docstring-Marker. Der Marker
gilt als „Default in Reichweite des Maintainers" beim Coden;
`_doc/ux.ini` ist die Übersetzungs- und Pflegeschicht.

## Schlüssel-Konventionen

UIDs sind die `name`-Werte aus `serverTypes[]`. In der Praxis:

- Klasse:  `gws.<bereich>.<modul>.<Modul>.<Klasse>`
  z. B. `gws.plugin.postgres.provider.Config`
- Property: `<Klassen-UID>.<property-Name>`
  z. B. `gws.plugin.postgres.provider.Config.host`
- VARIANT-UIDs entstehen synthetisch:
  `gws.ext.object.<kategorie>` (z. B. `gws.ext.object.layer`).

Bei Unsicherheit hilft ein Blick in eine frisch erzeugte
`app/__build/specs.json` — die UID ist exakt das, was dort als `name`
oder `uid` steht.

## Build & Verifikation

```
./make.sh spec               # Codegen + Spec-Lauf, schreibt nach app/__build/
python -m gws.spec.generator.coverage --lang de --threshold 80
                             # Zeigt Coverage pro Modul und gated CI
```

Tests für den Sammler liegen in `app/gws/spec/generator/_test/` und
laufen ohne Docker-Stack:

```
PYTHONPATH=app pytest app/gws/spec/generator/_test/
```

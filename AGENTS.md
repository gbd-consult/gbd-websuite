# Repository Guidelines

## Project Structure & Module Organization

GBD WebSuite is a plugin-based GIS web platform. Server-side Python code is under `app/gws/`, with framework modules in `app/gws/base/`, core utilities in `app/gws/core/`, GIS helpers in `app/gws/gis/`, and plugins in `app/gws/plugin/<name>/`. Client code is in `app/js/src/` and uses TypeScript, React, Redux, and OpenLayers. Generated build/spec output goes to `app/__build/`. Documentation sources live in `doc/books/`, demo configuration in `demos/`, and packaging/install tooling in `install/`.

Do not hand-edit generated files such as `app/gws/__init__.py` or `app/gws/ext/__init__.py`; update the corresponding `.pyinc` files or `app/gws/ext/types.txt`, then regenerate specs.

## Build, Test, and Development Commands

Use `./make.sh` as the main entry point:

- `./make.sh spec`: regenerate Python init files and specs in `app/__build/`.
- `./make.sh client`: build the production client bundle.
- `./make.sh client-dev`: build a development client bundle.
- `./make.sh client-dev-server`: start the client development server.
- `./make.sh doc`: build HTML documentation.
- `./make.sh test go`: start the Docker test stack, run tests, then stop it.
- `./make.sh clean`: remove generated build artifacts and bundles.

Most commands accept `--manifest <path>` or `GWS_MANIFEST` to scope generation and runtime configuration.

## Coding Style & Naming Conventions

Python formatting is governed by Ruff in `pyproject.toml`: 150-character lines and single quotes. `gws` is treated as first-party for import ordering. Keep Python test files named `*_test.py` or placed in existing `_test/` directories. Client files use `.ts`/`.tsx` for TypeScript and React; follow existing module names under `app/js/src/gc/`.

Plugin modules should stay under `app/gws/plugin/<plugin_name>/`. Extension classes declare `extName` values from `gws.ext.*`; update `app/gws/ext/types.txt` only when adding a new extension type.

## Testing Guidelines

Tests use pytest and are orchestrated by `app/gws/test/test.py` through `./make.sh test`. Use `./make.sh test go -k <expr>` to filter by test name, `./make.sh test go -o <regex>` to filter by filename, and `./make.sh test go -c` to generate coverage. Add focused tests near the code being changed, following the existing `_test.py` or `_test/*_test.py` patterns.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `fix typo` or `export function`. Keep commit subjects concise and specific. Pull requests should describe the change, mention any manifest or generated-spec impact, list test commands run, and include screenshots for visible client or documentation changes.

## Security & Configuration Tips

Do not commit local manifests, credentials, database dumps, or generated temporary data. Configuration commonly uses `.cx` files or JSON manifests; prefer explicit `--manifest <path>` during local testing.

## Konfigurator UX Strings

The WebSuite is the authoritative source for the help texts that the external Konfigurator UI renders next to each config option. Each module ships its UX docs alongside the code; the spec generator aggregates them into top-level `uxStrings` and `scenarios` blocks in `app/__build/specs.json`. Convention: `app/gws/spec/_doc/README.md`.

- `<module>/_doc/ux.ini` — `[de]`/`[en]` sections with `label`, `purpose`, `whenToUse`, `complexity`, `useCases`, `seeAlso`, `example` per class and property.
- `<module>/_doc/strings.ini` — older i18n strings (button labels, generic UI text).
- `<module>/_doc/scenarios.json` — apply-templates the Konfigurator offers as one-click starters; `template.type` must match an existing variant member.
- `app/gws/ext/_doc/ux.ini` and `app/gws/ext/_doc/scenarios.json` — central texts for VARIANT parents (Konfigurator picker headers before a member is chosen).
- Tools: `python -m gws.spec.generator.bootstrap_ux <module>` to generate skeletons; `python -m gws.spec.generator.coverage [--lang de] [--threshold 80]` to gate CI against new undocumented properties.

After editing any of these files, run `./make.sh spec` to refresh `specs.json` — that is what the Konfigurator consumes.

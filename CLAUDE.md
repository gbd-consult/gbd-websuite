# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

GBD WebSuite is a plugin-based, web-based open-source GIS platform: an application server, a web server, and a geo server in one codebase. The Python server lives under `app/gws/`, the TypeScript/React client under `app/js/`. QGIS integration is implemented as a regular plugin that talks to a QGIS Server instance.

Current version is in `app/VERSION`. Documentation: https://gbd-websuite.de/doc/8.0/index.html.

## Build & development (make.sh)

`make.sh` is the single entry point for all build/dev tasks. Every target that touches Python or client code first runs **codegen** (`app/_make_init.py` to assemble `gws/__init__.py` and `gws/ext/__init__.py` from `.pyinc` includes / `ext/types.txt`, then `app/gws/spec/spec.py` to generate specs into `app/__build/`). If you change a `.pyinc` file, a class with `gws.ext.*` extName, or the manifest, you must rebuild — running `./make.sh spec` is the cheapest way.

```
./make.sh spec                  # regenerate codegen + specs only
./make.sh client                # production client bundle
./make.sh client-dev            # dev client bundle
./make.sh client-dev-server     # webpack-style dev server
./make.sh doc                   # build docs (HTML)
./make.sh doc-markdown -out <dir>   # build docs as Markdown
./make.sh doc-api               # API reference
./make.sh image                 # docker images (via install/image.py)
./make.sh package               # tarball (via install/package.py)
./make.sh test <subcommand>     # see "Tests" below
./make.sh clean                 # remove __build and bundles
```

All targets accept `--manifest <path>` (or `GWS_MANIFEST` env var) to scope the build/spec to a manifest. Pass `-h` after any subcommand for its options.

## Tests

Tests run **inside docker compose** in a **two-stage** flow:
1. Host: `app/gws/test/test.py` reads `app/test.ini`, generates `docker-compose.yml`, brings up the stack (gws, qgis, postgres, mockserver, optional ldap).
2. Container: `app/gws/test/container_runner.py` runs pytest inside the gws container against the live services.

Service config (images, ports, postgres credentials, LDAP, mockserver) lives in `app/test.ini`; override by passing your own via `--ini` or `GWS_TEST_INI`. Test files are `*_test.py` (`pytest` with `--import-mode=importlib`).

```
./make.sh test go                       # configure → start → run → stop
./make.sh test start                    # bring stack up (leave running)
./make.sh test run                      # re-run tests in started stack (fast iteration)
./make.sh test stop
./make.sh test go -k <pytest-expr>      # filter by test name
./make.sh test go -o <regex>            # filter by filename
./make.sh test go -l                    # mount local app/ into the container (live edit)
./make.sh test go -c                    # produce coverage report
./make.sh test go - <pytest args>       # everything after `-` is forwarded to pytest
```

The runner mounts a working dir (`app/___test` by default) at the same path inside containers so host and container paths match. `service.gws.image` / `service.qgis.image` / `service.postgres.image` in `test.ini` pin the docker images used. `GWS_IN_TEST=1` is exported inside the test container — code can branch on it for test-only behavior.

## Architecture

### Plugin system via `gws.ext`

The plugin/extension model is the most important architectural concept. The list of extension types (action, application, layer, model, owsService, authProvider, etc.) lives in `app/gws/ext/types.txt`. `_make_init.py` reads this file and rewrites `app/gws/ext/__init__.py` so that every type is exposed as `gws.ext.object.<type>`, `gws.ext.config.<type>`, `gws.ext.props.<type>`, and `gws.ext.new.<type>`. Plugins declare classes by inheriting these tags (`extName = 'gws.ext.object.<type>'`), and command methods are tagged with `gws.ext.command.{api,cli,get,post,raw}`.

Plugins live in `app/gws/plugin/<name>/` (e.g. `qgis`, `postgres`, `alkis`, `ows_server`, `auth_provider/*`). The QGIS plugin is illustrative: it parses `.qgs` XML directly (no QGIS Python API), turns it into capability objects, and renders by proxying to QGIS Server.

### `__init__.pyinc` and `# @include`

`app/gws/__init__.py` is **generated**, not hand-edited. Its source is `app/gws/__init__.pyinc`, which uses `# @include relative/path.pyinc` directives that `_make_init.py` expands. Several `types.pyinc` files across `core/`, `lib/`, `gis/`, `base/*`, `server/`, `spec/` are pulled into the top-level `gws` namespace this way. To change a public type, edit the relevant `types.pyinc` and re-run codegen.

### Spec runtime

`app/gws/spec/` extracts type information from the source tree (`generator/`) and exposes it at runtime (`runtime.py`, `reader.py`) for config loading, CLI parsing, JSON-RPC, and TypeScript stub generation. `_gws.py` (the CLI) loads the manifest, creates a spec runtime, then dispatches via `gws.base.action.parse_cli_request` — CLI commands are `gws <cmd1> <cmd2>` translated to camelCase method names (e.g. `gws auth password` → `authPassword`).

### Web request dispatch

Outside CLI, the server has **one dynamic endpoint**: `/_` (underscore). All API calls go through it with a `cmd` parameter (`GET ?cmd=mapHttpGetBox&...`, slash-form `/_/mapGetBox/projectUid/...`, or `POST` with JSON body `{"cmd":"...", "params":{...}}`). `cmd` is split into action + method (e.g. `mapHttpGetBox` → action `map`, method `httpGetBox`). Plugin command methods are tagged with `gws.ext.command.{api,cli,get,post,raw}` and selected based on transport. Static files are served outside `/_` from per-site `web.sites[].root.dir`.

### MapProxy sidecar

The runtime stack includes **MapProxy** (pinned in `install/pip.lst`, exposed on a separate port — see `mpx_expose_port` in `test.ini`). It handles tile caching, scaling and reprojection of raster sources. When debugging raster pipelines, expect requests to leave gws and hit MapProxy before the upstream WMS/WMTS.

### Server entry points

- `app/bin/gws` is the shell wrapper used inside the container; sets `GWS_APP_DIR`/`GWS_VAR_DIR`/`GWS_TMP_DIR` and execs `app/bin/_gws.py`.
- `app/gws/server/` holds `cli.py`, `control.py`, `manager.py`, `monitor.py`, `uwsgi_module.py` — the HTTP server is uWSGI-based.
- `app/gws/base/` is the framework: `application/`, `auth/`, `action/`, `layer/`, `map/`, `model/`, `project/`, `printer/`, `ows/`, `template/`, `web/`, `database/`, etc.

### Client (`app/js/`)

`app/js/helpers/builder.js` is a **custom bundler** (not webpack/vite/rollup) invoked by `make.sh client*`. It reads specs from `app/__build/`, then produces `app.bundle.json` plus `vendor.bundle.js` and `util.bundle.js`. Source is in `app/js/src/` and the stack is intentionally pinned: **TypeScript 4.3, React 16.13, Redux 4.0, OpenLayers 4.6**. These are old by today's standards — when touching the client, do not assume modern tooling (no hooks-everywhere, no Vite HMR; use `client-dev-server` for live reload). Bundle filenames are mirrored in `app/gws/core/const.py` — keep both sides in sync. `tsconfig.json` defines path aliases `gws/*`, `@build/*`, and a fallback to `app/js/src/*` and `app/js/node_modules/*`.

### Configuration format

Configs are usually `.cx` (a templated config syntax — see `app/gws/lib/vendor/jump` and `demos/config_base.cx`) or JSON. `MANIFEST.json` enumerates which plugins are active and is read by both codegen and the runtime. Default search paths include `/data/MANIFEST.json` and `$GWS_MANIFEST`.

### In-tree vendor libraries

`app/gws/lib/vendor/` contains **in-tree libraries that are not on PyPI** — treat them as project code, not dependencies:
- `jump` — the `.cx` template engine (compiler + engine).
- `slon` — the SLON config DSL parser used for `.cx` configs.
- `jvv` — JSON value validator.
- `dog` — internal doc-generator backend used by `make.sh doc`.
- `umsgpack.py` — pure-python msgpack for the binary client/server protocol.

Plugins also follow a directory convention: `_doc/` (Markdown docs assembled by the doc builder), `_demo/` (demo configs picked up by `./make.sh demo-config`), `_test/` or `_test.py` (tests), and a `js/` subdir for plugin-specific TypeScript that gets bundled with the client.

### UX strings for the Konfigurator (`_doc/ux.ini`, `_doc/scenarios.json`)

The WebSuite is the **single source of truth** for the help texts the external Konfigurator UI shows next to each config option. Plugin maintainers contribute these texts as files alongside their module — the spec generator collects them into top-level `uxStrings` and `scenarios` blocks in `app/__build/specs.json` and `gws.generated.ts`. Convention is documented in `app/gws/spec/_doc/README.md`.

- **`<module>/_doc/ux.ini`** — per-module structured UX docs. Has `[de]` and `[en]` sections; keys follow `<dotted.class.path>[.<property>].<field>`. Allowed fields: `label`, `purpose`, `whenToUse`, `complexity`, `useCases`, `seeAlso`, `example`. Class-level entries describe whole config classes; property-level entries describe individual fields. Example: `app/gws/plugin/postgres/_doc/ux.ini`.
- **`<module>/_doc/strings.ini`** — older i18n strings (button labels, generic UI text). Lives next to `ux.ini`; both are loaded.
- **`<module>/_doc/scenarios.json`** — apply-templates the Konfigurator offers as one-click starters (e.g. „Standard PostgreSQL provider", „LDAP gegen AD"). Format: `{ "<uid>": [ { "title": {"de": ..., "en": ...}, "purpose": {...}, "template": {"type": "...", ...} } ] }`. The `template.type` discriminator must match an existing variant member.
- **VARIANT parents** — synthetic `gws.ext.config.<type>` UIDs (the families a Konfigurator picker shows before a member is chosen) live centrally in `app/gws/ext/_doc/ux.ini` and `app/gws/ext/_doc/scenarios.json`. They have no `tProperties`, only class-level texts.
- **Field-list markers in class docstrings** — optional, lower priority than `ux.ini`. Recognized: `:complexity:`, `:seeAlso:`, `:since:`, `:deprecated:`.

Tooling under `app/gws/spec/generator/`:

```
python -m gws.spec.generator.bootstrap_ux <module>      # generate ux.ini skeleton from existing specs
python -m gws.spec.generator.coverage [--lang de] [--threshold 80] [--manifest <path>]
```

`coverage.py` is the recommended CI guard against new undocumented config properties — it exits non-zero when any module drops below `--threshold`. Pytest tests for the collectors live in `app/gws/spec/generator/_test/` and run without Docker (`PYTHONPATH=app pytest app/gws/spec/generator/_test/`).

After editing any `_doc/ux.ini`, `_doc/strings.ini`, `_doc/scenarios.json` or `ext/types.txt`, re-run `./make.sh spec` to refresh `app/__build/specs.json`. The generated `specs.json` is what the Konfigurator consumes — once it is wired up, the legacy `ux-schema.json` in the Konfigurator repo is retired.

## Code conventions

- **Python**: ruff with `line-length = 150`, `quote-style = 'single'`, ignores `F541`. isort treats `gws` as first-party. mypy config in `app/mypy.ini` is loose (`follow_imports = skip`, missing-imports ignored) — it is a soft check, not a gate.
- **Generated files are committed**: `app/gws/__init__.py` (~4800 lines) and `app/gws/ext/__init__.py` are generated **and** checked into git. After editing a `.pyinc` or `ext/types.txt`, run codegen and commit both source and generated file in the same change.
- **Test discovery**: files named `*_test.py`; pytest options are baked into `app/test.ini`.
- **Bundle constants**: `JS_BUNDLE`, `JS_VENDOR_BUNDLE`, `JS_UTIL_BUNDLE` must match between `app/js/helpers/builder.js` and `app/gws/core/const.py`.
- **Branches**: development happens on `master`; release branches follow `r<major>.<minor>` (e.g. `r8.0`, `r8.2`, `r8.3`). Backports are merged from `master` into the relevant release branch.

## Installation (reference only)

The production deployment is the docker image `gbdconsult/gws-server:<tag>`; see `INSTALL.md`. There is also an experimental Debian/Ubuntu install script at `install/install.sh` that targets `/var/gws` by default.

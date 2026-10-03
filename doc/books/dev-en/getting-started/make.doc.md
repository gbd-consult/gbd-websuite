# Building with make.sh :/dev-en/getting-started/make

`make.sh` in the repository root runs all build tasks:

```
make.sh [-d] <command> [--manifest <path>] <command options>
```

`make.sh <command> -h` shows the options of a command.

## Commands ::

| Command | Task |
|---------|------|
| `spec` | generate the specs, see [](/dev-en/server/specs) |
| `client` | build the production client |
| `client-dev` | build the development client |
| `client-dev-server` | start the client development server on port 8080 |
| `test` | run the tests, see [](/dev-en/server/testing) |
| `doc` | build the documentation |
| `doc-api` | build the API documentation |
| `doc-markdown` | build the documentation as Markdown |
| `doc-dev-server` | start the documentation development server on port 5500 |
| `demo-config` | compile the configuration for the demo projects |
| `image` | build a docker image |
| `package` | copy the application, without build artifacts, to a directory |
| `clean` | remove all build artifacts |

## Code generation ::

Every command except `clean` and `demo-config` first runs code generation:

1. `app/_make_init.py` generates `app/gws/ext/__init__.py` from `app/gws/ext/types.txt` and `app/gws/__init__.py` from `app/gws/__init__.pyinc`.
2. `app/gws/spec/spec.py` generates the specs into `app/__build`.

`make.sh spec` runs only this step.

## Options ::

`--manifest <path>` passes a manifest to the code generation step, so that the specs and the client build include its plugins. Without it, the `GWS_MANIFEST` environment variable is used. The option must follow the command name immediately.

`-d` (`--docker`) runs the command in the GWS docker image instead of the host, with the working copy mounted at the same path. This avoids installing Python libraries on the host. It works for `spec`, `doc`, `doc-api`, `doc-markdown`, `doc-dev-server`, `demo-config` and `package`.

The interpreters can be set with the `GWS_PYTHON` and `GWS_NODE` environment variables.

## Build artifacts ::

| Path | Contents |
|------|----------|
| `app/__build/specs.json` | specs |
| `app/__build/gws.generated.ts` | TypeScript types for the client |
| `app/__build/configref.*.md` | configuration reference |
| `app/__build/doc/<version>` | documentation |
| `app/__build/apidoc/<version>` | API documentation |
| `app/__build/doc_markdown/<version>` | documentation as Markdown |
| `app/*.bundle.*`, `<plugin>/app.bundle.json` | client bundles |

# Setup :/dev-en/getting-started/setup

## Requirements ::

The server runs in a Docker container, so the host needs only a few tools:

- Docker with the Compose plugin
- Bash and Python 3, to run `make.sh`
- Node.js, only to build the client

The code generation step of `make.sh` uses the Python standard library only. Commands that need more libraries, such as the documentation build, can run inside the GWS image with `make.sh -d`, see [](/dev-en/getting-started/make).

## Working copy ::

Clone the repository:

```
git clone https://github.com/gbd-consult/gbd-websuite
```

Top-level directories:

| Directory | Contents |
|-----------|----------|
| `app` | the application: server code in `app/gws`, client code in `app/js`, CLI scripts in `app/bin` |
| `data` | default `/data` directory of the docker image, with a demo configuration |
| `demos` | shared configuration and assets for the demo projects |
| `doc` | documentation sources, theme and build script |
| `install` | docker image and package builders, lists of system and Python packages |

Build artifacts go to `app/__build`. Files and directories starting with `___` are ignored by the build tools and can be used for local experiments.

## IDE ::

Add `app` to the Python path of your IDE, so that the `gws` package can be resolved. For VS Code with Pylance:

```json
{
    "python.analysis.extraPaths": ["/path/to/gbd-websuite/app"]
}
```

For a working copy of a plugin outside of this repository, add the same path. Linter and formatter settings are in `pyproject.toml` (ruff) and `app/mypy.ini`.

`app/gws/__init__.py` and `app/gws/ext/__init__.py` are generated, but committed. Run `make.sh spec` after changing a `.pyinc` file or `app/gws/ext/types.txt`, see [](/dev-en/server/layout).

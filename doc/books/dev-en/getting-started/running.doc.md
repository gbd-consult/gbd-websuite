# Running a development server :/dev-en/getting-started/running

## Docker image ::

The server runs in the image `gbdconsult/gws-amd64` or `gbdconsult/gws-arm64`, tagged with the release, for example `gbdconsult/gws-amd64:8.5`. The image contains the system libraries, a Python environment in `/opt/venv` and a copy of the application in `/gws-app`. Its default command is `gws server start`.

For development, mount your working copy of `app` over `/gws-app`. The container then runs your code, and the image provides only the runtime.

## Directories ::

| Container path | Purpose |
|----------------|---------|
| `/gws-app` | application code, mount `gbd-websuite/app` here |
| `/data` | configuration, project files and other user data |
| `/gws-var` | persistent server data: caches, the stored configuration, generated server configs |
| `/tmp/gws` | temporary data, cleared on server start |

The image's `/data` contains a demo configuration, the same as `data` in the repository.

## Compose file ::

A minimal `docker-compose.yml`:

```yaml
services:
    gws:
        image: gbdconsult/gws-amd64:8.5
        container_name: gws-container
        ports:
            - "3333:80"
        volumes:
            - /path/to/gbd-websuite/app:/gws-app
            - /path/to/my-project/data:/data
            - /path/to/gws-var:/gws-var
        tmpfs:
            - /tmp
        environment:
            - GWS_CONFIG=/data/config.cx
            - GWS_LOG_LEVEL=DEBUG
            - GWS_UID=1000
            - GWS_GID=1000
```

Start it with `docker compose up` and open `http://localhost:3333`. The client application needs compiled bundles, which are not part of the working copy. Build them once with `make.sh client`.

## Environment variables ::

| Variable | Default | Meaning |
|----------|---------|---------|
| `GWS_CONFIG` | first of `/data/config.cx`, `.json`, `.yaml`, `.py` | path to the main configuration file |
| `GWS_MANIFEST` | `/data/MANIFEST.json` | path to the application manifest |
| `GWS_LOG_LEVEL` | `INFO` | log level, overrides `server.log.level` |
| `GWS_WEB_WORKERS` | `4` | number of web workers, overrides `server.web.workers` |
| `GWS_SPOOL_WORKERS` | `4` | number of background workers, overrides `server.spool.workers` |
| `GWS_UID`, `GWS_GID` | `1000` | user and group the server runs as |
| `GWS_VAR_DIR` | `/gws-var` | persistent data directory |
| `GWS_TMP_DIR` | `/tmp/gws` | temporary directory |

Set `GWS_UID` and `GWS_GID` to your host user, so that files written to mounted directories belong to you. The variables are defined in <% pyapi('gws.core.env') %>.

## Configuration ::

The configuration is written in `cx`, JSON, YAML or Python. `cx` is the native format: a template language on top of SLON, a JSON-like notation without commas and colons. A minimal configuration:

```
permissions.read "allow all"

actions+ { type "project" }

projects+ {
    uid "hello"
    title "Hello"
}
```

All options are described in the [configuration reference](/admin-de/reference). Check a configuration without starting the server:

```
docker exec gws-container gws server configtest
```

## Manifest ::

The manifest is an optional JSON file that lists external plugins and application-wide build options. Lines starting with `//` or `#` are comments.

```json
{
    "plugins": [
        {"path": "/data/plugins/my_plugin"}
    ],
    "withStrictConfig": true
}
```

| Key | Meaning |
|-----|---------|
| `plugins` | external plugins, each with a `path` and an optional `name` (defaults to the directory name) |
| `tsConfig` | `tsconfig.json` for the client build |
| `withStrictConfig` | refuse to start if the configuration has errors |
| `withFallbackConfig` | start with a minimal built-in configuration if the configuration fails |

Plugin paths are container paths, the plugin directory must be mounted.

## Applying changes ::

The server configures itself once on start and stores the configured object tree in `/gws-var`. Web and background workers load the stored tree.

| Change | Command |
|--------|---------|
| configuration files | none, the server reconfigures automatically |
| Python code | `docker exec gws-container gws server reconfigure` |
| a `.pyinc` file or `types.txt` | `make.sh spec`, then `gws server reconfigure` |

`gws server reload` restarts the workers without configuring.

With the developer option `server.auto_reload`, uWSGI restarts the workers whenever a Python module changes. The workers then load the stored tree with the new code, so changes to command methods take effect immediately. Changes to `configure` and to `Config` classes still need `gws server reconfigure`. Developer options are described in [](/dev-en/server/debugging):

```
developer {
    "server.auto_reload" true
}
```

## CLI ::

`gws` is the command line interface of the server, available in the container's `PATH`. `gws -h` lists all commands, `gws <command> -h` shows the options of a command. Commands have two parts, for example:

```
docker exec gws-container gws server configtest
docker exec gws-container gws auth password
docker exec gws-container gws cache status
```

`-v` turns on debug logging for a single command.

# Debugging :/dev-en/server/debugging

## Logging ::

Set `GWS_LOG_LEVEL=DEBUG` in the container environment, or `server.log.level "DEBUG"` in the configuration. At the debug level, every log line shows its source file and line. CLI commands log at the debug level with `-v`.

Log with <% pyapi('gws.core.log') %>, available as `gws.log`:

```py
gws.log.debug(f'loading {path=}')
gws.log.exception()
```

<% pyapi('gws.core.log.exception', 'gws.log.exception()') %> logs the current exception with its chain of causes.

<% pyapi('gws.core.debug') %>, available as `gws.debug`, has helpers for quick inspection. They log at the debug level:

| Function | Purpose |
|----------|---------|
| <% pyapi('gws.core.debug.p', 'gws.debug.p(obj, d=3)') %> | log the structure of an object, up to the depth `d` |
| <% pyapi('gws.core.debug.p', 'gws.debug.p(stack=True)') %> | log the call stack |
| <% pyapi('gws.core.debug.time_start', 'gws.debug.time_start(label)') %>, <% pyapi('gws.core.debug.time_end', 'gws.debug.time_end()') %> | log the time between the two calls |

## Developer options ::

The `developer` section of the configuration turns on debugging features. Write keys with dots in quotes:

```
developer {
    "request.log_all" true
    "db.engine_echo" true
}
```

| Option | Effect |
|--------|--------|
| `request.log_all` | write every request to `/gws-var/debug` |
| `db.engine_echo` | log all SQL statements |
| `server.auto_reload` | restart the uWSGI workers when a Python module changes |
| `template.always_reload` | reload templates on every use |
| `template.raise_errors` | raise template errors instead of logging them |
| `web.reload_bundles` | reload client bundles on every request |

The server logs a warning on start if developer options are set.

## Object inspector ::

The `admin` action provides an HTML page to browse the configured object tree of the running server. Add the action to the configuration:

```
actions+ { type "admin" }
```

Log in as a user with the `admin` role and open `/_/adminInspector`. The page shows the root and lets you follow attributes, dict keys and list items of every object. The `path` parameter addresses an object directly, starting with a node uid, for example `/_/adminInspector?path=my_layer/provider`. The `search` parameter lists objects with a property containing a text, or a specific property with `prop=text`.

The same action provides a viewer for cached tiles at `/_/adminMapCache`.

## Configuration dumps ::

When the server reads a configuration file, it writes intermediate results to `/gws-var/config`: the rendered `cx` template (`.src.slon`), the parsed file (`.src.json`) and the validated configuration (`.parsed.json`). Use them to check how templates and includes were resolved.

## Scripts ::

Code under uWSGI is hard to step through. Reproduce a problem in a script instead: configure an application, then send requests to it without the web server.

```py
import werkzeug.test

import gws
import gws.config
import gws.base.web.wsgi_app

gws.u.ensure_system_dirs()

cr = gws.config.configure(config_path='/data/config.cx')
gws.config.log_report(cr)
root = gws.activate_root(cr.root)

client = werkzeug.test.Client(gws.base.web.wsgi_app.make_application(root))
res = client.post('/_/projectInfo', json={'projectUid': 'hello'})
print(res.status, res.get_json())
```

Run it in the container as the server user:

```
docker exec gws-container gws -p /data/debug.py
```

`root` gives access to the whole tree, for example <% pyapi('gws.Root.get', "root.get('my_layer')") %> returns the node with the uid `my_layer`. The functions for configuring and loading are in <% pyapi('gws.config.loader') %>.

## Debugger ::

To step through a script with VS Code, install `debugpy` in the container, publish a port and start the script under the debugger:

```
docker exec gws-container pip install debugpy
docker exec -it --user 1000:1000 --env PYTHONPATH=/gws-app gws-container python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client /data/debug.py
```

Attach from VS Code with a `launch.json` configuration that maps the container paths to your working copy:

```json
{
    "name": "Attach to GWS",
    "type": "debugpy",
    "request": "attach",
    "connect": {"host": "localhost", "port": 5678},
    "pathMappings": [
        {"localRoot": "/path/to/gbd-websuite/app", "remoteRoot": "/gws-app"}
    ]
}
```

The port 5678 must be published in the compose file. Use the `GWS_UID` and `GWS_GID` of the server for `--user`.

# Testing and debugging :/dev-en/debug

The following sections describe GWS debug and test frameworks.


## Host debugging

You can debug server-side GWS code directly on the host machine, without running any servers or docker containers, provided required libraries and python modules are installed, e.g. in a virtualenv. See `install/apt.lst` and `install/pip.lst` for the list of requirements.

In order to use the GWS App it has to be configured and activated. This can be done with the following calls:

```py

import gws.config

config_path = 'path to your config'
manifest_path = 'path to your manifest'

root = gws.config.configure(
    config_path=config_path,
    manifest_path=manifest_path
)

gws.config.activate(root)
```

Now, the `root` object contains the configured GWS object tree. You can access the `Application` object with `root.app` or find an object of interest with `root.get`. For example, if you about to debug an object with an id `my_object`:

```py

my_obj = root.get('my_object')
my_obj.method_to_be_debugged()
```

To make web requests to the API, you need the `werkzeug.client` object, to which you pass `wsgi_app.initialized_application`. Here's a complete example where we invoke `projectInfo` for the project `test`, provided it is configured in your config file.

```py

import werkzeug

import gws.config
import gws.base.web.wsgi_app

config_path = 'my_config.cx'
manifest_path = 'my_manifest.json'

root = gws.config.configure(
    config_path=config_path,
    manifest_path=manifest_path
)

gws.config.activate(root)

client = werkzeug.test.Client(
    gws.base.web.wsgi_app.initialized_application,
    werkzeug.wrappers.Response
)

request = {
    "params": {
        "projectUid": "test",
        "localeUid": "de_DE"
    }
}
response = client.open(method='POST', path='/_/projectInfo', json=request)
print(response.json)
```


## Container debugging


In order to observe the GBD WebSuite in it's natural habitat, the container, we can leverage debugpy and the DAP (Debug Adapter Protocol). This allows every editor/IDE that supports DAP to attach to a running python process and set breakpoints and inspect the application state.

I create a folder `~/gws/debug` that contains all required boilerplate:

- `docker-compose.debug.yaml`
- `server.sh`
- `wsgi_main.py`
- `gws`

### docker-compose.debug.yml


```yaml
services:
    qgis:
        image: gbdconsult/gbd-qgis-server-amd64:3.34.12
        container_name: qgis
        # ... see other composefiles for missing settings here

    gws:
        image: gbdconsult/gws-amd64:8.1
        container_name: gws
        ports:
            - "0.0.0.0:3333:80" # default http on 3333
            - "0.0.0.0:5000:5000" # forward mpx port (optional)
            - "0.0.0.0:5678:5678" # debug adapter port
        volumes:
            - ${GWS_PROJECT_DIR}/gws-${GWS_PROJECT_NAME}/data:/data:ro
            - ${GWS_VAR_DIR}:/gws-var
            - ${GWS_PROJECT_DIR}/gbd-websuite/app:/gws-app

            # this is only relevant for client-side plugin development
            - /plugins:/plugins

            # this allows us to inject debug capabilities into the container
            - .:/debug
        # here we call our custom start script
        command: ["/debug/gws", "server", "start"]
        tmpfs:
            - /tmp
        environment:
            - GWS_CONFIG=/data/config/local.cx
            - GWS_MANIFEST=/data/MANIFEST.json
            - GWS_LOG_LEVEL=DEBUG
            - PG_SERVICEFILE=/data/pg_service.local.conf
            # these might fix problems when starting/attaching to debug server
            - GWS_WEB_WORKERS=1
            - PYDEVD_LOAD_NATIVE_LIB=0
            - PYDEVD_USE_CYTHON=0
```

### uwsgi_web.ini

This will replace the `/gws-var/server/uwsgi_web.ini` with a debugfriendly version, where we enforce only one thread, and don't kill our worker after 60s

Ensure that you make the following changes, rest stays the same:

```ini
[uwsgi]
daemonize = false
threads = 1
# harakiri-verose = true
# harakiri = 60
processes = 1
wsgi-file = /debug/wsgi_main.py
honour-stdin = true
single-interpreter = true
```

I'm not 100% sure if honour-stdin and single-interpreter are neccessary, needs testing.


### server.sh

This will replace `/gws-var/server.sh`, as this is where uwsgi is started, and we need to pass the path to our `uwsgi_web.ini`

```sh
rsyslogd -i /tmp/gws/pids/rsyslogd.pid -f /gws-var/server/syslog.conf
uwsgi --ini /debug/uwsgi_web.ini
uwsgi --ini /gws-var/server/uwsgi_mapproxy.ini
uwsgi --ini /gws-var/server/uwsgi_spool.ini
exec nginx -c /gws-var/server/nginx.conf
```

### wsgi_main.py

This is the entry point of the application for uwsgi. We can start the debug server in this file, which allows us to attach to the application from our IDE:

```py
import debugpy

# the in_process_debug_adapter=True is important because of uwsgi shenaningans
# the 0.0.0.0 ip is important, because of container port forwarding
debugpy.listen(("0.0.0.0", 5678), in_process_debug_adapter=True)

# if we want to immediatly hold and wait until we manually attach from our ide
# we can add
# debugpy.wait_for_client()

import gws.base.web.wsgi_app as wsgi_app

application = wsgi_app.application
```

We could also add the debugpy related lines to any other python file, for example inside a plugin which only purpose is to enable debugging, but this way we have the option
to wait for debug adapter client attaching, and set a breakpoint during import of gws.base.web.wsgi_app

### gws

This replaces the `/gws-app/bin/gws` script, and is the file we define as entry command in our docker-compose.yml. Modify the original script as follows:

First thing we want to do is ensure debugpy is installed

```sh
#!/bin/bash
pip install debugpy

...
```


To debug the configuration step, replace the following lines:

```sh
$PYTHON $MAIN_PY "$@"
```
with
```sh
$PYTHON -m debugpy --listen 0.0.0.0:5678 --wait-for-client $MAIN_PY "$@"
```


To debug the running application behind uwsgi point the `exec` calls to our `/debug/server.sh` which handles starting the uwsgi and nginx processes.


### Running the Container

just like any other container:

`docker compose -f docker-compose.debug.yml up -d qgis`

and

`docker compose -f docker-compose.debug.yml up gws`

depending on if you have told debugpy to wait-for-client during configuration, you will have to connect before the application starts, and then connect again when the uwsgi app is running


### Attaching from VSCode

in your `.vscode/launch.json` you can create a configuration as follows:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}/app",
                    "remoteRoot": "/gws-app"
                }
            ]
        }
    ]
}
```

ensure you have correct path mappings in order to be able to set breakpoints
you can set a breakpoint before attaching, to halt instantly after a `debugpy.wait_for_client()` line

#### connecting from different workspace (plugin directory) in vscode

if you have a plugin as it's own workspace you can add a path mapping like

```json
{
    "localRoot": "${workspaceFolder}",
    "remoteRoot": "/plugins/temporal"
},
{
    "localRoot": "/home/<user>/gws/gbd-websuite/app",
    "remoteRoot": "/gws-app"
}
```

in order to debug the plugin code, and also be able to jump into gws functions

in that case you might also want to add the following extra paths in your `.vscode/settings.json`

```json
{
    "python.autoComplete.extraPaths": [
        "/home/<user>/gws/gbd-websuite/app/"
    ],
    "python.analysis.extraPaths": [
        "/home/<user>/gws/gbd-websuite/app/"
    ],
}
```

and add a `tsconfig.json` like this to your workspace root for typescript support

```json
{
    "extends": "../../gbd-websuite/app/js/tsconfig.json",
}
```

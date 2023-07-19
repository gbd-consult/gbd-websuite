## Interactive debugging :/dev-en/debugging

You can debug server-side GWS code interactively in your IDE, without starting any servers or docker containers, provided required libraries and python modules are installed on the host, e.g. in a virtualenv. See `install/apt.lst` and `install/pip.lst` for the list of requirements.

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

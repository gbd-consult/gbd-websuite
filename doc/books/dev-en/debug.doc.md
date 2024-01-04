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

## Server Test Framework

In order to run tests in GWS, you need to start a GWS container and some auxiliary containers, and run the test script in the GWS container. All test functionality is invoked via `make.sh test` commands. 

There are two ways to run the tests:

- automatically, with `make.sh test go`. This configures the docker-compose environment for testing, runs tests and shuts down.
- manually, where you first start the environment with `make.sh test start` and then invoke tests with `make.sh test run` in another shell

In both cases, you can select specific tests with the `--only` option and provide additional pytest options. See `make.sh test -h` for details.

A couple of examples:

```shell
# automatic test of everything
/path/to/gws/root/make.sh test go 

# automatic test of module 'foo;
/path/to/gws/root/make.sh test go --only foo 

# start the test framework
/path/to/gws/root/make.sh test start 

# (in a new shell) run tests for 'foo'
/path/to/gws/root/make.sh test run --only foo
```

### Configuration

The configuration for tests is in `test.ini` in the application root directory. If you need custom options (e.g. local directory names), create a secondary `ini` file with your overrides and pass it as `--ini myconfig.ini`.

### Test files

All test files must end with `_test.py`, all test functions must start with `test_`. It is recommended to always import the test utilities library, which provides some useful shortcuts and mocks. Here is an example of a test file:

```py

"""Testing the foo package."""

import gws
import gws.test.util as u
import gws.lib.foo as foo

def test_one():
    assert foo.bar == 1

def test_two():
    with u.raises(ValueError):
        foo.blah()
```

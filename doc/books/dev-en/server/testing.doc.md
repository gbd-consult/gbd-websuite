# Testing :/dev-en/server/testing

Tests use pytest and run in a docker compose environment with the GWS image, PostgreSQL/PostGIS, QGIS Server, an LDAP server and a mock HTTP server.

## Running tests ::

`make.sh test go` starts the test environment, runs all tests and stops it again. While working on code, start the environment once with `make.sh test start`, run selected tests with `make.sh test run` as often as needed, and stop it with `make.sh test stop`. Options select test files and tests, mount the working copy, and turn on coverage and debug logging. Options after `-` are passed to pytest. `make.sh test -h` lists all options.

## Configuration ::

Defaults are in `app/test.ini`. Put local overrides in your own ini file and pass it with `--ini` or the `GWS_TEST_INI` environment variable:

```ini
[runner]
base_dir = /path/to/test-base

[service.gws]
image = gbdconsult/gws-arm64:8.5

[service.qgis]
image = gbdconsult/gbd-qgis-server-arm64:3.40
```

`base_dir` holds the generated compose file, the test configuration, the PostgreSQL data and the coverage report. Relative paths are resolved against the ini file.

## Writing tests ::

Tests are in `_test` directories next to the code, in files named `*_test.py`. The runner finds them in the whole `gws` package, there is no central test directory.

Test utilities are in `gws.test.util`, imported as `u`. A test that configures an application from an inline configuration and calls a command:

```py
import gws
import gws.test.util as u


@u.fixture(scope='module')
def root():
    u.pg.create('plain', {'id': 'int primary key', 'name': 'text', 'g': 'geometry(point,3857)'})
    u.pg.insert('plain', [
        dict(id=1, name='a11', g=u.pg.ewkb('POINT(10 100)')),
    ])
    cfg = '''
        permissions.all "allow all"
        actions+ { type 'map' }
        projects+ {
            uid "A"
            map.layers+ {
                uid "LAYER"
                type "postgres"
                tableName "plain"
                templates+ { subject "feature.label" type "html" text "--{id}/{name}--" }
            }
        }
    '''
    yield u.gws_root(cfg)


def test_get_features(root: gws.Root):
    res = u.http.get(root, '/_/mapGetFeatures/layerUid/LAYER')
    assert res.json['features'][0]['views']['label'] == '--1/a11--'
```

## Utilities ::

| Function | Purpose |
|----------|---------|
| `u.gws_root(cfg, **vars)` | configure and activate a root from a `cx` string; `{name}` placeholders are replaced with `vars` |
| `u.http.get(root, url)`, `u.http.post(root, url)` | send a request to the application |
| `u.http.api(root, cmd, request)` | call an API command |
| `u.pg.create(table, columns)`, `u.pg.insert(table, rows)`, `u.pg.rows(sql)` | prepare and query the test database |
| `u.auth.add_user(name, password, roles)` | add a user to the mock authorization provider |
| `u.mockserver.add(snippet)` | add a handler to the mock HTTP server |
| `u.option(name)` | read an option of the test configuration, for example service hosts and ports |
| `u.fixture`, `u.raises` | `pytest.fixture`, `pytest.raises` |

The database configured by `u.gws_root` is the PostgreSQL service of the test environment. The full list of utilities is in <% pyapi('gws.test.util') %>.

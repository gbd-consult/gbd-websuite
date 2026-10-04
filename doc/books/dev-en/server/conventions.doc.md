# Code conventions :/dev-en/server/conventions

## Formatting ::

Code is formatted with ruff, settings are in `pyproject.toml`: indents of 4 spaces, single quotes, lines up to 150 characters.

## Imports ::

Import modules, not names, and use qualified names:

```py
import gws.lib.osx

gws.lib.osx.run(cmd)
```

Exceptions are `typing` names (`from typing import Optional, cast`) and relative imports within a package (`from . import provider`).

Group imports in this order, separated by blank lines:

1. `typing`
2. standard library
3. third-party libraries
4. `import gws`, then other `gws` modules
5. relative imports

## Naming ::

| Kind | Style | Example |
|------|-------|---------|
| classes | CapWords | `FeatureProps` |
| fields of data classes, object attributes | camelCase | `tableName`, `self.serverMgr` |
| functions, methods, local variables | snake_case | `create_child`, `layer_uid` |
| modules | snake_case | `auth_provider` |
| constants | UPPER_CASE | `MAX_BOX_SIZE` |
| commands | camelCase | `mapGetFeatures` |
| module-private names | leading underscore | `_read_file` |

Configuration keys, request parameters and props are camelCase, because they are shared with the client and the configuration files.

## Modules ::

A module that implements an extension type defines `Config`, `Props` and `Object` under these names. Interfaces from `gws` are implemented by a class with the same name in the implementing package, for example <% pyapi('gws.base.feature.Feature', 'gws.base.feature.Feature') %> implements <% pyapi('gws.Feature', 'gws.Feature') %>.

## Types ::

Annotate the fields of data classes, they define the specs. Annotate object attributes at the class level and function signatures where the types are not obvious. Use `Optional[T]` for optional fields of data classes, the spec generator uses it to determine defaults.

## Docstrings ::

Docstrings follow the [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). Start every module with a docstring. Field docstrings follow the field:

```py
class Config(gws.Config):
    """Search provider."""

    url: gws.Url
    """Service url."""
```

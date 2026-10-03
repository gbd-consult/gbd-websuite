# Plugins and extension types :/dev-en/server/plugins

## Extension types ::

An extension type is a set of classes registered under a category and a type name. For example, the PostgreSQL layer is the type `postgres` of the category `layer`:

```py
@gws.ext.config.layer('postgres')
class Config(gws.base.database.layer.Config):
    """Layer that shows features from a PostgreSQL table."""


@gws.ext.object.layer('postgres')
class Object(gws.base.database.layer.Object):
    ...
```

| Decorator | Registers |
|-----------|-----------|
| `gws.ext.object.<category>('<type>')` | the object class |
| `gws.ext.config.<category>('<type>')` | the configuration class |
| `gws.ext.props.<category>('<type>')` | the props class, used for client types |

A type that is created from the configuration needs a registered `Config`. A registered `Props` class is needed only if the props differ from the base class.

The decorated classes get two attributes: `extName`, the full name like `gws.ext.object.layer.postgres`, and `extType`, the type name like `postgres`.

## Categories ::

The categories are listed in `gws/ext/types.txt`. An extension implements the interface of its category, usually by extending a base class from `gws.base`:

| Category | Interface | Purpose |
|----------|-----------|---------|
| `action` | <% pyapi('gws.Action') %> | actions that provide commands |
| `application` | <% pyapi('gws.Application') %> | the application |
| `authMethod` | <% pyapi('gws.AuthMethod') %> | how credentials arrive: login form, HTTP basic, token |
| `authMultiFactorAdapter` | <% pyapi('gws.AuthMultiFactorAdapter') %> | second authentication factors: e-mail, TOTP |
| `authProvider` | <% pyapi('gws.AuthProvider') %> | user sources: file, database, LDAP |
| `authSessionManager` | <% pyapi('gws.AuthSessionManager') %> | session storage |
| `cli` | <% pyapi('gws.Node') %> | CLI commands |
| `databaseProvider` | <% pyapi('gws.DatabaseProvider') %> | database connections |
| `exporter` | <% pyapi('gws.Exporter') %> | feature export formats |
| `finder` | <% pyapi('gws.Finder') %> | search sources |
| `helper` | <% pyapi('gws.Node') %> | application-wide services |
| `layer` | <% pyapi('gws.Layer') %> | map layers |
| `legend` | <% pyapi('gws.Legend') %> | legend sources |
| `map` | <% pyapi('gws.Map') %> | maps |
| `model` | <% pyapi('gws.Model') %> | data models |
| `modelField` | <% pyapi('gws.ModelField') %> | model field types |
| `modelValidator` | <% pyapi('gws.ModelValidator') %> | field validators |
| `modelValue` | <% pyapi('gws.ModelValue') %> | computed field values |
| `modelWidget` | <% pyapi('gws.ModelWidget') %> | client widgets for fields |
| `owsProvider` | <% pyapi('gws.OwsServiceProvider') %> | OWS data sources |
| `owsService` | <% pyapi('gws.OwsService') %> | OWS services: WMS, WMTS, WFS, CSW |
| `printer` | <% pyapi('gws.Printer') %> | printers |
| `project` | <% pyapi('gws.Project') %> | projects |
| `storageProvider` | <% pyapi('gws.StorageProvider') %> | backends for the client storage |
| `template` | <% pyapi('gws.Template') %> | template types |

To add a category, add it to `types.txt` and run `make.sh spec`.

## Using extension types ::

In the configuration, a list or a property of an extension category takes objects with a `type` key:

```
models+ { type "postgres" tableName "streets" }
```

In `Config` classes, such properties are declared with `gws.ext.config.<category>`. In code, the object is created with the category as class reference, and the `type` selects the class:

```py
class Config(gws.Config):
    models: Optional[list[gws.ext.config.model]]
    """Data models."""


class Object(gws.Node):
    def configure(self):
        self.models = self.create_children(gws.ext.object.model, self.cfg('models'))
```

Helpers (category `helper`) are application-wide services, configured once in the `helpers` list. Get them with `self.root.app.helper('<type>')`.

## Registration ::

The decorators do nothing at runtime. The spec generator finds them in the source code, see [](/dev-en/server/specs). So an extension type is available as soon as its module is in a package under `gws`, or in an external plugin listed in the manifest. The module is imported only when the type is used.

## Example ::

A complete validator plugin, `gws/plugin/model_validator/regex/__init__.py`:

```py
"""Regex validator for strings."""

import re

import gws
import gws.base.model.validator


@gws.ext.config.modelValidator('regex')
class Config(gws.base.model.validator.Config):
    """Checks that a value matches a regular expression."""

    regex: gws.Regex
    """Regular expression, matched anywhere in the value unless anchored."""


@gws.ext.object.modelValidator('regex')
class Object(gws.base.model.validator.Object):
    regex: str

    def configure(self):
        self.regex = self.cfg('regex')

    def validate(self, field, feature, mc):
        val = feature.attributes.get(field.name)
        if not isinstance(val, str):
            return False
        return re.search(self.regex, val) is not None
```

## CLI commands ::

A CLI command is a method of a class in the category `cli`. The class extends `gws.Node`, the method takes one parameter that extends <% pyapi('gws.CliParams') %>:

```py
import gws
import gws.config


class CountParams(gws.CliParams):
    projectUid: str
    """Project uid."""


@gws.ext.object.cli('hello')
class Object(gws.Node):

    @gws.ext.command.cli('helloCount')
    def count(self, p: CountParams):
        """Count the layers of a project."""

        root = gws.config.load()
        ...
```

The command name `helloCount` is called as `gws hello count -projectUid ...`. The method docstring is the help text, the parameter fields are the options. CLI objects are not part of the object tree, the command loads the stored configuration itself if it needs it. See <% pyapi('gws.server.cli') %> for an example.

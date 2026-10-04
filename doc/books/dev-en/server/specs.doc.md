# Specs :/dev-en/server/specs

The specs describe all types the server exchanges with the outside world. They are generated from the source code by <% pyapi('gws.spec.generator') %> and used at runtime by <% pyapi('gws.spec.runtime') %>.

## What is collected ::

The generator parses the Python modules of `gws` and of the plugins in the manifest, without importing them. It collects:

- extension classes, from the `gws.ext.object`, `gws.ext.config` and `gws.ext.props` decorators
- commands, from the `gws.ext.command` decorators, with their parameter and return types
- all classes, enums and type aliases reachable from these
- docstrings of classes, fields and enum members
- translations from `strings.ini` files

## Where specs are used ::

The server generates the specs each time it configures, and the `gws` CLI each time it runs, so code changes need no build step. The specs are used to:

- import extension classes when the configuration uses them
- validate and convert configurations
- validate and convert request parameters
- list CLI commands and their options

`make.sh spec` writes the specs to `app/__build` for the development tools:

| File | Used for |
|------|----------|
| `specs.json` | client build |
| `gws.generated.ts` | client types, see [](/dev-en/server/actions) |
| `configref.en.md`, `configref.de.md` | configuration reference, see [](/dev-en/documentation/reference) |

## Supported types ::

Fields of <% pyapi('gws.Config') %>, <% pyapi('gws.Props') %>, <% pyapi('gws.Request') %> and <% pyapi('gws.Response') %> classes can use:

- `str`, `int`, `float`, `bool`, `bytes`, `Any`
- `list[T]`, `set[T]`, `dict`, `dict[str, T]`, `tuple[...]`
- `Optional[T]`, `Literal[...]`
- enums extending <% pyapi('gws.Enum') %>
- other data classes
- extension variants like `gws.ext.config.layer`, which accept any type of the category

Union types are not supported.

## Special types ::

Some types are strings in the configuration and are converted when the configuration is read:

| Type | Configuration value | Runtime value |
|------|--------------------|----------------|
| <% pyapi('gws.AclStr') %> | `"allow admin, deny all"` | <% pyapi('gws.Acl') %> |
| <% pyapi('gws.CrsName') %> | `"EPSG:3857"`, `3857` | <% pyapi('gws.Crs') %> |
| <% pyapi('gws.Duration') %> | `"1h30m"`, `90` | seconds as `int` |
| <% pyapi('gws.DateStr') %>, <% pyapi('gws.DateTimeStr') %> | ISO date or date-time | `datetime` |
| <% pyapi('gws.FilePath') %>, <% pyapi('gws.DirPath') %> | path, relative to the configuration file | absolute path, must exist |
| <% pyapi('gws.UomValueStr') %> | `"10px"`, `"5mm"` | <% pyapi('gws.UomValue') %> |
| <% pyapi('gws.UomPointStr') %>, <% pyapi('gws.UomSizeStr') %> | `"10px 20px"` | <% pyapi('gws.UomPoint') %>, <% pyapi('gws.UomSize') %> |
| <% pyapi('gws.UomExtentStr') %> | four values and a unit | <% pyapi('gws.UomExtent') %> |
| <% pyapi('gws.Regex') %> | regular expression | `str`, checked for syntax |
| <% pyapi('gws.Url') %> | `http` or `https` URL | `str` |

<% pyapi('gws.Node.cfg', 'self.cfg()') %> returns the converted values. The conversions are defined in <% pyapi('gws.spec.reader') %>.

## Docstrings ::

Docstrings of `Config` classes and their fields end up in the configuration reference. Enum members are documented the same way as fields:

```py
class DisplayMode(gws.Enum):
    """Layer display mode."""

    box = 'box'
    """Display a layer as one big image."""
    tile = 'tile'
    """Display a layer in a tile grid."""
```

A docstring that ends with `(added in 8.1)`, `(changed in 8.2)` or `(deprecated in 8.3)` is shown with a version label.

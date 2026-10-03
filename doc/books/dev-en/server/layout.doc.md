# Code layout :/dev-en/server/layout

## The gws package ::

`app` is the Python path root, `app/gws` is the package. Its subpackages are described in [](/dev-en/overview/components). Packages under `gws.base` are always loaded. Packages under `gws.plugin` and external plugins are loaded only when the configuration uses them.

## Basic types ::

`gws/__init__.py` holds the basic types and the interfaces of all components, so that every module can use them after `import gws`. The file is generated from `gws/__init__.pyinc`, which includes `types.pyinc` files from the packages with `# @include` lines:

```
# @include core/_data.pyinc
# @include base/layer/types.pyinc
```

An interface is a class whose methods have only docstrings, for example <% pyapi('gws.Layer') %>. The package that implements it, like `gws.base.layer`, defines a class that extends the interface. To add or change an interface, edit the package's `types.pyinc` and run `make.sh spec`.

Short aliases are available everywhere: `gws.u` (<% pyapi('gws.core.util') %>), `gws.c` (<% pyapi('gws.core.const') %>), `gws.log`, `gws.debug` and `gws.env`.

## Package contents ::

A typical package, like `gws/base/map`:

| Path | Contents |
|------|----------|
| `__init__.py` | package exports, often the main `Config` and `Object` |
| `*.py` | modules, one per extension or concern |
| `types.pyinc` | interfaces included in `gws/__init__.py` |
| `_doc/strings.ini` | translations of configuration docstrings |
| `_doc/<book>/*.doc.md` | documentation sections for a book, for example `_doc/admin-de` |
| `_demo/*.cx` | demo projects |
| `_test/*_test.py` | tests |
| `js/` | client code of the package |

Plugins, in `gws/plugin` or external, have the same layout, but no `types.pyinc`: `gws/__init__.py` includes only interfaces of the core packages, and plugins implement them.

The spec generator skips files whose path contains `test`, `___`, `/vendor/` or `__pycache__`.

## External plugins ::

An external plugin is a directory outside of the repository, listed in the manifest. It has the same layout as a package in `gws/plugin`. Its modules are imported with the plugin directory as the top-level package, so modules within a plugin import each other with relative imports:

```py
from . import provider
```

The parent directory of a plugin must not contain an `__init__.py`. For deployment, `make.sh package <dir> -manifest <path>` copies the plugins listed in the manifest into `gws/plugin`.

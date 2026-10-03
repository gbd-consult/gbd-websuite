# Custom documentation :/dev-en/documentation/custom

The GWS documentation build can also build documentation for your projects and plugins. Write the documentation in `*.doc.md` files, create an options file in JSON and pass it to `make.sh doc`:

```
make.sh doc -opt /path/to/my-options.json
```

The options file can contain any Dog option, see `app/gws/lib/vendor/dog/options.py`. Relative paths in `docRoots` and `extraAssets` are resolved against the options file. `docRoots` replaces the default (the GWS repository), `extraAssets` is added to the default list.

## Standalone documentation ::

To build a separate site, list only your directory in `docRoots` and write a root section `:/`:

```json title="my-options.json"
{
    "docRoots": ["/path/to/my/docs"],
    "title": "My Project"
}
```

```md title="index.doc.md"
# My Project :/

Welcome!
```

## Adding to the GWS documentation ::

To publish your documentation together with the GWS books, list both your directory and the GWS repository, and give your root section a SID under `/extra`:

```json title="my-options.json"
{
    "docRoots": ["/path/to/my/docs", "/path/to/gbd-websuite"]
}
```

```md title="index.doc.md"
# My Project :/extra/my-project

Welcome!
```

The root section of the GWS documentation embeds all `/extra/*` sections, so your documentation appears as another book.

Single options can be overridden on the command line with `-D<option> <value>`, for example `-DwebRoot /docs`.

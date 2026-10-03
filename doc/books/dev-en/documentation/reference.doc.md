# Configuration reference :/dev-en/documentation/reference

The [configuration reference](/admin-de/reference) is generated from the source code. It lists every class reachable from the application `Config`, with its properties, types, defaults and docstrings.

## Generation ::

The spec generator writes the reference as Markdown to `app/__build/configref.en.md` and `app/__build/configref.de.md`. The administrator book includes the German version with `%include`. Each class gets its own section, whose SID is the class name, for example `gws.base.map.core.Config`. The `%ref` command links to these sections.

`make.sh doc` and the documentation development server regenerate the reference before building.

## Translations ::

English texts are the docstrings in the code. Translations are in `strings.ini` files in the `_doc` directory of each package, for example `app/gws/plugin/select_tool/_doc/strings.ini`:

```ini
[de]
gws.plugin.select_tool.action.Config = Aktion zum Auswählen von Features in der Karte.
gws.plugin.select_tool.action.Config.storage = Speicher zum Sichern und Laden von Auswahlen.
```

The key is the full name of a class, a class and a property, or an enum and a member. Sections are languages. Lines starting with `;`, `#` or `//` are comments, `\n` is a line break.

If a translation is missing, the reference shows the English text.

## Finding keys ::

Add `?dev=1` to the URL of the reference page to show the key of every text. Missing translations are marked with `???`. To change a text, search the project for its key, or add it to the `strings.ini` of the package that defines the class.

# Overview :/dev-en/documentation/overview

The documentation is written in Markdown and built with Dog, a generator that is part of this repository (`gws.lib.vendor.dog`). The build script is `doc/doc.py`, its options are in `doc/options.py`, and the theme is in `doc/theme`.

## Books ::

The documentation consists of books, one per audience and language:

| Book | Section | Language |
|------|---------|----------|
| `doc/books/common-de` | `/common-de` | German |
| `doc/books/user-de` | `/user-de` | German |
| `doc/books/admin-de` | `/admin-de` | German |
| `doc/books/dev-en` | `/dev-en` | English |

`doc/books/index.doc.md` is the root section `/`, which links and embeds the books.

## Sources ::

Dog collects all `*.doc.md` files in the repository, except paths matching `node_modules`, `___` or `__build`. File names and locations do not matter: each section declares its position with a section id (SID) in its heading, and a section is included where another section embeds its SID. Dog is described in [](/dev-en/documentation/dog).

So the documentation of a component can be kept next to its code, in `_doc/<book>` directories. For example, `app/gws/base/map/_doc/admin-de/map.doc.md` defines the section `/admin-de/konfiguration/map`, which is embedded in the administrator book.

Section files are also [Jump](https://github.com/gebrkn/jump) templates. Besides Dog's own commands, the GWS build provides these commands, defined in `doc/extra_commands.cx.html`:

| Call | Output |
|------|--------|
| `%ref "gws.base.map.core.Config"` | link to a class in the configuration reference |
| `%demo "select_tool"` | link to a demo project |
| `<% '<' + '% pyapi("gws.base.map.core.Object") %' + '>' %>` | link to a Python name in the API documentation |

`pyapi` takes a fully qualified name of a module, class, function or attribute, and shows its last component as the link text. An optional second argument sets a different link text, for example `<% '<' + '% pyapi("gws.Node.cfg", "self.cfg()") %' + '>' %>`.

## Creating a book ::

1. Create a directory in `doc/books`, named `<topic>-<language>`, for example `doc/books/admin-en`.
2. Add an `index.doc.md` with the root section of the book:
    ```
    # Administrator Guide :/admin-en

    ## :/admin-en/introduction
    ```
3. Embed the book in `doc/books/index.doc.md` and add a link to it:
    ```
    - [Administrator Guide](/admin-en)

    ## :/admin-en
    ```
4. Optionally, set how the book is split into HTML files in `fileSplitLevel` in `doc/options.py`. The default is 3: sections up to three SID components deep, like `/admin-en/topic/subtopic`, get their own pages, deeper sections are included in the page of their parent.

## Building ::

| Command | Output |
|---------|--------|
| `make.sh doc` | HTML in `app/__build/doc/<version>` |
| `make.sh doc -pdf` | HTML and a PDF |
| `make.sh doc-markdown` | Markdown in `app/__build/doc_markdown/<version>` |
| `make.sh doc-api` | API documentation in `app/__build/apidoc/<version>` |
| `make.sh doc-dev-server` | development server, see [](/dev-en/documentation/dev-server) |

All URLs in the HTML start with the option `webRoot`, `/doc/<version>` by default. The API documentation is built with Sphinx from the docstrings, its configuration is in `doc/api`. `pyapi` links point to `<webRoot>/api`, so publish the API documentation in the `api` directory of the HTML output.

Building the PDF requires `wkhtmltopdf`, which is installed in the GWS image. Use `make.sh -d doc` to build in the image.

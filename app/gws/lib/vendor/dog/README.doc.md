# Dog, the documentation generator :/dev-en/documentation/dog

Dog collects documents in a given root folder (source tree), compiles them into a tree of nested sections, and generates a set of HTML files or a PDF from this tree.

Since the docs are "scattered" across the source tree, each module or plugin can keep its docs within its own folder.

Document filenames and paths do not matter; the structure of the documentation is determined solely by document headings.

The docs are written in Markdown plus [Jump](https://github.com/gebrkn/jump) commands.

## Structure

Documentation in Dog is a nested hierarchy of "sections". A _section_ is a chunk of text with a _heading_ and a _SID_ (section id). A SID is like a path in a virtual documentation filesystem.

Each section starts with a Markdown heading, optionally followed by a colon and a SID.

```md
# Introduction :/docs/developer/intro
```

Deeper headings (with more `#`) create the section hierarchy.

```md
# First Top-Level Section

## Subsection 1

## Subsection 2

# Second Top-Level Section

## Subsection

### Sub-sub-section
```

**The number of `#`s in a heading does not set the output level.** Headings only build the section hierarchy. The rendered level (`h1`, `h2`, …) of a section is derived from its depth in that hierarchy — the number of slashes in its SID — together with the _file split-level_ (an option that controls how sections are grouped into files, see "Options"). How many `#`s a heading has is irrelevant to the output.

**Give the topmost heading in each file a single `#`.** Because the `#` count carries no meaning, matching it to the section's depth gains nothing; starting every file at `#` keeps the source uniform wherever its sections land in the tree.

### Section embedding

A section can be "embedded" (included) in another section by writing its SID in a heading. The embedded section's content is inserted at that point.

For example, this heading:

```md
# :/docs/developer/intro
```

inserts the content of the section `/docs/developer/intro` in place of the heading; the heading itself is not rendered.

**Note that a section must be embedded somewhere, or it will be dropped.** Dog starts at the root section `/` and keeps only the sections reachable from it through embedding. A section that is written but never embedded ("unbound" section) is dropped from the output. You'll get a warning if you have unbound sections in your docs.

Assume our documentation is structured like this:

```md title="index.doc.md"
# Our docs :/docs

## Welcome

Hi there

## :/docs/first

## More stuff

More text

## :/docs/second
```

```md title="first.doc.md"
# First Thing :/docs/first

Discussion of the first thing
```

```md title="second.doc.md"
# Second Thing :/docs/second

Discussion of the second thing
```

Then the compiled documentation will look like this:

```md title="result.md"
# Our docs

## Welcome

Hi there

## First Thing

Discussion of the first thing

## More stuff

More text

## Second Thing

Discussion of the second thing
```

### Embedding multiple sections

An embedded section SID can contain a wildcard component `*`, in which case all matching sections are included and sorted alphabetically by their titles. For example, if we have these files:

```md title="index.doc.md"
# Our docs :/docs

## Details

Some details:

### :/docs/details/*
```

```md title="tags.doc.md"
# Tags :/docs/details/tags

Something about tags
```

```md title="attributes.doc.md"
# Attributes :/docs/details/attributes

Something about attributes
```

```md title="values.doc.md"
# Values :/docs/details/values

Something about values
```

The result will be like this:

```md title="result.md"
# Our docs

## Details

Some details:

### Attributes

Something about attributes

### Tags

Something about tags

### Values

Something about values
```

### More on SIDs

A section SID denotes the section's position in the tree and is used to refer to it from elsewhere. A SID consists of _components_ separated by slashes; a component may contain letters (any case), digits, dots, dashes and underscores — for example `gws.base.auth.manager.Config`.

Every heading becomes a section, and every section gets a _final SID_ — an absolute path such as `/docs/intro`. It is worked out from the heading's annotation (the text after the colon, if any), its title, and its position — the `#` level, which fixes the section's parent (the nearest section one level up) and its previous sibling (the last section at the same level). Where a SID has to be generated from a title, the title is lowercased and reduced to letters, digits and dashes, so "Beta Stuff" becomes `beta-stuff`.

An **absolute** annotation — one starting with a slash, like `:/docs/intro`, or the bare root `:/` — is the section's final SID as written. A **relative** annotation like `:intro` or `:sub/thing`, and a heading with **no annotation at all**, are anchored to the section's surroundings: the SID is added to the parent section, or, for a top-level heading that has no parent, it replaces the last component of the previous top-level section. When there is no annotation, the component generated from the title takes the place of the relative SID.

A **trailing slash** means "and then append the generated component": `:more/` is resolved as a relative (or absolute) SID and the title's component is added after it, so `:more/` on a heading titled "Fine Print" under `/docs/gamma` yields `/docs/gamma/more/fine-print`.

Finally, a top-level heading that has a title but neither an annotation nor a previous section becomes the root `/` — this is how the first heading of a standalone document is chosen as the root. Because relative SIDs need something to anchor to, the first section in a file must resolve to an absolute SID. All final SIDs are normalized, so redundant `.`, `..` and doubled slashes collapse.

<!-- prettier-ignore -->
```md
# First Section :/docs/first

This section has a complete absolute SID.

## Alpha Stuff :alpha

This subsection has a relative SID, which is added to the parent.
The final SID of this section will be `/docs/first/alpha`

## Beta Stuff

This subsection has no SID,
so a SID will be generated and added to the parent.
The final SID will be `/docs/first/beta-stuff`

## Gamma Stuff :gamma

This subsection has a relative SID, which is added to the parent.

### Delta Stuff :delta

This section is deeper than "gamma",
so "gamma" will be the parent.
The final SID will be `/docs/first/gamma/delta`

### Epsilon Stuff

This section is deeper than "gamma" and there's no SID,
so a generated SID will be added to the parent.
It will be `/docs/first/gamma/epsilon-stuff`

### Fine Print :more/

This section is deeper than "gamma" and its SID ends with a slash,
so a generated SID will be appended to it.
The final SID will be `/docs/first/gamma/more/fine-print`.
Note that this section becomes a 4th level section,
despite the 3rd level heading.

# Second Thing

This section is top-level and has no SID,
so it will replace the last component of the previous section's SID
with a generated SID: `/docs/second-thing`.
```

Within an output file, a section's HTML anchor is built from its SID components joined with a dash. Because a dash is also legal inside a component, two different SIDs in the same file can produce the same anchor — for example `/docs/data-types` and `/docs/data/types` both give `data-types`. Plain headings (`::`) take an anchor from their title in the same way, and can clash with each other or with a section anchor on the same page. Dog does not rename clashing anchors automatically; give one of them a different SID or title.

### Plain headings

A heading marked with a double colon `::`, in place of a SID, is not a section. It is rendered as an ordinary heading within the body of the enclosing section — useful for headings that should structure a page visually without appearing in the navigation or splitting off into their own files.

```md
## Configuration :/config

Some intro text.

### Examples ::

More text, still part of the `/config` section.
```

Unlike a section heading, a plain heading renders at its literal Markdown level (`###` becomes an `h3`). It gets an anchor id derived from its title for deep-linking, and it does not take part in the SID resolution of the sections around it.

### Section linking

A section can be linked to with its SID in the standard Markdown link notation:

```md
See [](/docs/first/second) for more details

See also [here](/docs/first/third)
```

If no text is given, the section title will be used. Relative SIDs are resolved relative to the containing section's SID.

Linking to a section doesn't make it part of the tree. The section still needs to be embedded somewhere.

### Working with assets

You can refer to any asset (image, video, document) from the source tree just by mentioning its filename, no matter where the file is physically located:

```md
Image: Look at ![this](picture.jpg)

Link: See [our price list](prices.pdf)
```

If you have several assets with the same filename, provide just enough of the path to tell them apart:

```md
Look at ![this](color/picture.jpg)

Look at ![this](bw/picture.jpg)
```

A link or image target is resolved as an asset if it matches a known asset filename, otherwise as a section SID. In practice the file extension is what tells them apart: assets have one (`picture.jpg`, `prices.pdf`), SIDs normally do not (`/docs/intro`, `intro`). External URLs (`http:`, `https:`) are left as-is.

## Markdown extensions

Dog provides a few extensions to standard Markdown.

### Tables

Tables work as in [GFM](https://github.github.com/gfm/):

    | foo | bar |
    |-----|-----|
    | baz | bim |

_Output:_

| foo | bar |
| --- | --- |
| baz | bim |

### Autolinks

Everything that looks like a URL is converted to a link:

    > Our markdown formatter is https://mistune.lepture.com

_Output:_

> Our markdown formatter is https://mistune.lepture.com

### Code blocks

Fenced code blocks are formatted with [Pygments](https://pygments.org/):

    ```py
    print("Hi", 40 + 2)  # test
    ```

_Output:_

```py
print("Hi", 40 + 2)  # test
```

Additionally, you can add a title and line numbers to the block. `numbers=N` turns on line numbers and starts counting at `N`:

    ```py title="Example 1" numbers=5
    print("Hi", 40 + 2)  # line 5
    spam()
    ham()
    eggs()
    ```

_Output:_

```py title="Example 1" numbers=5
print("Hi", 40 + 2)  # line 5
spam()
ham()
eggs()
```

### Decorations

A decoration looks like `{myclass text}` and generates an HTML `span` element with the class name `md-decoration-myclass`.

    > Click the {button Exit} button to exit

_Output:_

> Click the {button Exit} button to exit

### Link attributes

Attributes can be set on links and images, similarly to [Pandoc](https://pandoc.org/MANUAL.html#extension-link_attributes). `width` and `height` attributes accept arbitrary CSS units. This extension currently works for inline elements only.

    > Some image ![](logo.svg){.someclass .otherclass width=3em height=20px border=1}

_Output:_

> Some image ![](logo.svg){.someclass .otherclass width=3em height=20px border=1}

## Commands

Dog supports all Jump commands (like `if` or `include`) and provides a set of its own commands. To avoid excessive escaping, Jump syntax is redefined as follows:

    %quote xmp
    #% commands start with a percent sign
    %include foo

    #% echoes are enclosed in <% %>
    <% someVar %>
    %end xmp

### toc

Creates a local table of contents for the given section ids. If `depth` is omitted, it defaults to 1. A section's own `tocDepth` (see "Options") still caps how deep its subtree is listed. Relative SIDs are resolved relative to the container. You can also use `*`, as in section embedding.

    %quote xmp
    %toc depth=3
        /docs/first/thing
        /docs/second/thing
        /docs/misc/*
    %end
    %end xmp

### info

Creates an "info" admonition:

    %quote xmp
    %info
        To whom it may concern.
    %end
    %end xmp

_Output:_

<!-- prettier-ignore -->
%info
    To whom it may concern.
%end

### warn

Creates a "warning" admonition:

    %quote xmp
    %warn
        Here be dragons.
    %end
    %end xmp

_Output:_

<!-- prettier-ignore -->
%warn
    Here be dragons.
%end

### see

Creates a "see also" admonition:

    %quote xmp
    %see
        See also: [the docs](https://example.com/docs).
    %end
    %end xmp

_Output:_

<!-- prettier-ignore -->
%see
    See also: [the docs](https://example.com/docs).
%end

### graph

Draws a graph with [Graphviz](https://graphviz.org). The `dot` command must be installed and on your `PATH`. A diagram can have an optional caption.

    %quote xmp
    %graph "Simple graph"
        digraph {
            rankdir="LR"
            one -> two
        }
    %end
    %end xmp

_Output:_

<!-- prettier-ignore -->
%graph "Simple graph"
    digraph {
        rankdir="LR"
        one -> two
    }
%end

### dbgraph

Draws a diagram of a database schema. To describe the schema, use the syntax `table (columns)`, where `columns` is a comma-separated list of column definitions. A column definition consists of a name, an optional type, and an optional "pk" indicator (primary key) or an arrow `->` pointing to another table column (foreign key).

    %quote xmp
    %dbgraph "Our database"
        street (
            id pk,
            name text,
            city_id -> city.id
        )
        city (
            id pk,
            name text,
            population bigint
        )
        river (
            id pk,
            name text,
            length int
        )
        river_to_city (
            river_id -> river.id,
            city_id -> city.id
        )
    %end
    %end xmp

_Output:_

<!-- prettier-ignore -->
%dbgraph "Our database"
    street (
        id pk,
        name text,
        city_id -> city.id
    )
    city (
        id pk,
        name text,
        population bigint
    )
    river (
        id pk,
        name text,
        length int
    )
    river_to_city (
        river_id -> river.id,
        city_id -> city.id
    )
%end

## Python API

Dog provides the following API functions:

```python
dog.build_html(options: Options | dict)
```

builds the HTML documentation.

```python
dog.build_markdown(options: Options | dict)
```

builds the Markdown documentation.

```python
dog.build_pdf(options: Options | dict)
```

builds the PDF documentation (requires [wkhtmltopdf](https://wkhtmltopdf.org/)).

```python
dog.start_server(options: Options | dict)
```

starts a development server with live reload.

The `options` argument is either an object like the one below or a dictionary with the same keys and value types.

## Options

```python
%include ../dog/options.py
```

##### docRoots ::

The list of directories Dog scans for source files.

##### docPatterns ::

Shell globs that select documentation files. Default: `*.doc.md`.

##### assetPatterns ::

Shell globs that select asset files (images, PDFs, and so on). Default: `*.svg`, `*.png`.

##### excludeRegex ::

If set, any source path matching this regular expression is skipped — handy for `node_modules`, build directories and the like.

##### outputDir ::

The directory the generated files are written to.

##### webRoot ::

A prefix added to every URL Dog emits, so the site can be served from a subpath such as `/docs`.

##### staticDir ::

The subdirectory, under the web root, that holds shared assets and the search index. Default: `_static`.

##### extraAssets ::

A list of files — theme CSS and JS, logos, and so on — copied verbatim into the static directory.

##### tocDepth ::

A dictionary mapping a SID to how deep its subtree is shown in the sidebar and in `%toc`, counting the section itself. `1` shows the section but none of its children; a section not listed shows its whole subtree. For example, to collapse the reference chapter in the navigation:

    "tocDepth": {
        "/admin/reference": 1
    }

##### blendChars ::

Characters that split _and_ join words in the search index. With `-` a blend character, `buffer-size` is found as `buffer`, `size`, or the joined `buffer-size`, and the dash is kept in snippets. A good set for code-heavy docs is `-/._&`. Empty by default.

##### extraChars ::

Characters treated as part of a word in the search index — they neither split nor join. With `<` and `>` as extras, `<h1>` is a single searchable token. Empty by default.

##### serverHost ::

The host `start_server` binds to. Default: `0.0.0.0`.

##### serverPort ::

The port `start_server` listens on. Default: `5500`.

##### pdfPageTemplate ::

A page template used instead of `pageTemplate` when building a PDF with `build_pdf`.

##### pdfOptions ::

A dictionary of extra [wkhtmltopdf](https://wkhtmltopdf.org/) command-line options — margins, footer, and so on — for `build_pdf`.

##### fileSplitLevel ::

A dictionary that maps a SID to a _split-level_ — a number telling Dog how to distribute sections into output files. A split-level applies to the section itself and all its descendants, so the root `/` sets the default for the whole tree.

`0` (the default) puts all documentation in a single `index.html`. `1` creates one file per top-level section, keeping its subsections in the same file:

    /a      ->  /a/index.html
    /a/b    ->  /a/index.html
    /a/b/c  ->  /a/index.html
    /a/d    ->  /a/index.html

    /x/y    ->  /x/index.html
    /x/z    ->  /x/index.html
    /x/z/w  ->  /x/index.html

`2` gives level-one and level-two sections their own files:

    /a      ->  /a/index.html

    /a/b    ->  /a/b/index.html
    /a/b/c  ->  /a/b/index.html

    /a/d    ->  /a/d/index.html

    /x/y    ->  /x/y/index.html

    /x/z    ->  /x/z/index.html
    /x/z/w  ->  /x/z/index.html

and so on. You can override the default for any subtree. For example, to split everything at level two but keep the `/x` subtree in a single file:

    "fileSplitLevel": {
        "/": 2,
        "/x": 1
    }

    /a      ->  /a/index.html
    /a/b    ->  /a/b/index.html
    /a/b/c  ->  /a/b/index.html
    /a/d    ->  /a/d/index.html

    /x/y    ->  /x/index.html
    /x/z    ->  /x/index.html
    /x/z/w  ->  /x/index.html

##### pageTemplate ::

A [Jump](https://github.com/gebrkn/jump) template rendered for each HTML page. It receives these arguments:

| argument      | meaning                                       |
| ------------- | --------------------------------------------- |
| `args`        | the `pageTemplateArgs` dict                   |
| `breadcrumbs` | array of tuples `(section-url, section-head)` |
| `home`        | URL of the documentation home page            |
| `main`        | main HTML content for this page               |
| `options`     | the options object                            |
| `path`        | HTML path for this page                       |

##### pageTemplateArgs ::

Additional data for use in the page template, available there as `args`.

##### includeTemplate ::

A Jump template included at the top of every source file — the place to define custom Jump commands shared across the docs.

##### title ::

The documentation title, available to the page template as `options.title`.

##### subTitle ::

The documentation subtitle, available to the page template as `options.subTitle`.

##### debug ::

Enables verbose logging.

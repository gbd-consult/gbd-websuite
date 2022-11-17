# Dog, the documentation generator :/dev-en/documentation/dog

Dog collects documents in a given root folder (source tree), compiles them into a
tree of nested sections and generates a set of html files or a pdf from this tree.

Since the docs are "scattered" across the source tree, each module or plugin can keep its docs within its own folder.

Document filenames and paths do not matter, the structure of the documentation is determined solely by document headers.

The docs are written in Markdown + [Jump](https://github.com/gebrkn/jump) commands. 

## Structure

The Dog documentation is a nested hierarchy of "sections". A _section_ is a chunk of text with a _header_ and a _sid_ (section id).

Each section starts with a Markdown header, optionally followed by a colon and a sid.

    # Introduction :/docs/developer/intro

Deeper headings (with more dashes) create the section hierarchy.

    # First Top-Level Section
        ## Subsection 1
        ## Subsection 2
    # Second Top-Level Section
        ## Subsection
            ### Sub-sub-section

Heading levels are only used to create hierarchies, the final output rendering (`h1`, `h2` etc.) is determined by the section depth (the number of slashes) and the html split-level.

### Section embedding

A heading with no title and only a sid means "find a section with this sid and paste its content right here".

For example, assume we have files like this:

`index.doc.md`:

    # Our docs :/docs
    
    ## Welcome
        Hi there
    
    ## :/docs/first
    
    ## More stuff
        Some text
    
    ## :/docs/second

`first.doc.md`:

    ## First Thing :/docs/first
        Discussion of the first thing

`second.doc.md`:

    ## Second Thing :/docs/second
        Discussion of the second thing

Then, the compiled documentation will be like:

    # Our docs :/docs
    
    ## Welcome :/docs/welcome
        Hi there
    
    ## First Thing :/docs/first
        Discussion of the first thing
    
    ## More stuff :/docs/more-stuff
        Some text
    
    ## Second Thing :/docs/second
        Discussion of the second thing

An embedded section sid can contain a wildcard component `*`, in which case all matching sections are included and sorted alphabetically by their titles. For example, if we have these files:

`one.doc.md`:

    # Tags :/docs/details/tags    
        Something about tags

`two.doc.md`:

    # Attributes :/docs/details/attributes        
        Something about attributes

`three.doc.md`:

    # Values :/docs/details/values        
        Something about values

`index.doc.md`:

    # Our docs :/docs

    ## Details
        
    Some details:

    ## :/docs/details/*

The result will be like this:

    # Our docs :/docs

    ## Details :/docs/details
        
    Some details:

    ## Attributes :/docs/details/attributes        
        Something about attributes

    ## Tags :/docs/details/tags    
        Something about tags

    ## Values :/docs/details/values        
        Something about values

When Dog compiles your documentation, it starts with the section named `/` (root section) and recursively collects all embedded sections. Sections that are not embedded anywhere are formatted and generated, but won't become a part of the main documentation tree. 

### More on Sids

A section sid denotes the section position in the tree and is used to refer to the section from elsewhere.

A sid consists of _components_, separated by a slash. A component name can contain lowercase letters, digits, dots and
dashes.

A section can have an absolute sid (which starts with a slash), a relative one, or no sid at all. The final sid for a section is determined using the following algorithm:

- if the sid is `/`, it is taken as is. The section becomes the root section
- if no sid is given, the section title is converted to a component name and used as a sid
- if a sid is given and ends with a slash, the title is converted to a component name and appended to the sid
- if the sid is absolute, it is taken as is, and its heading level is ignored
- if the sid is relative, the compiler uses its heading level to locate parent and sibling sections in the current file
- if there is a parent section, the sid is added to the parent sid
- if there is a sibling section, the sid replaces the last component of the sibling sid
- if none of the above applies, error

The first section in a file must provide a sid and this sid has to be absolute.

Example:

    # First Section :/docs
        This section has a complete absolute sid.

        ## Alpha Stuff :alpha
            This sub-section has a relative sid, which is added to the parent.
            The final sid of this section will be `/docs/alpha`

        ## Beta Stuff
            This sub-section has no sid, so it will be generated and added to the parent.
            The final sid will be `/docs/beta-stuff`

        ## Gamma Stuff :gamma

            ### Some Details :details
                This section is deeper than "gamma", so "gamma" will be the parent.  
                The final sid will be `/docs/gamma/details`
        
            ### Fine Print :more/
                This section is deeper than "gamma" and its sid ends with a slash, so a generated sid will be added.
                The final sid will be `/docs/gamma/more/fine-print`.
                Note that this section becomes a 4th level section, despite the 3rd level heading.
        
    # Second Section :second
        This section has a relative sid, which replaces the last component of the sibling (`docs`).
        The final sid will be `/second`

    # Third Section
        This section has no sid, so it will be generated and merged with the sibling.
        The final sid will be `/third-section`

### Section linking

A section can be linked to with its sid in the standard Markdown link notation:

    See [](/docs/first/second) for more details

    See also [here](/docs/first/third)

If no text is given, the section title will be used. Relative sids are resolved relatively to the containing section sid.

Linking to a section doesn't make it a part of the tree. The section still needs to be embedded somewhere.

### Working with assets

You can refer to any asset (image, video, document) from the source tree just by mentioning its filename, no matter where
the file is physically located:

    Image: Look at ![this](picture.jpg)

    Link:  See [our price list](prices.pdf)

If you have multiple different assets with the same filename, provide just enough of the path to make a distinction:

    Look at ![this](color/picture.jpg)

    Look at ![this](bw/picture.jpg)

## Commands

Dog supports all Jump commands (like `if` or `include`) and provides a set of its own commands. To avoid excessive escaping, Jump syntax is redefined as
follows:

    %quote xmp
    #% commands start with a percent sign
    %include foo
    
    #% echoes are enclosed in <% %>
    <% someVar %>
    %end xmp


### toc

Creates a local table of contents for given section ids. If `depth` is omitted, it defaults to 1. Relative sids are
resolved relative to the container. You can also use `*` just like in the section embedding.

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

    %warn
        Here be dragons.
    %end

### graph

Draws a graph with [GraphViz](https://graphviz.org). The `dot` command must be installed and be in your `PATH`. A diagram can have an optional caption.


    %quote xmp
    %graph 'Simple graph'
        digraph {
            rankdir="LR"
            one -> two
        }
    %end
    %end xmp

    %graph 'Simple graph'
        digraph {
            rankdir="LR"
            one -> two
        }
    %end

### dbgraph

Draws a database diagram. 

A DB diagram consists of tables and arrows. A table is a name, followed by a list of columns in `(...)`. Each column has a name, an optional type and an optional key indicator (`pk` for a primary key, `fk` for a foreign key). An arrow is like `table.column -> table.column`, where `->` indicates an m:1 relation and `->>` an 1:m one.

    %quote xmp    
    %dbgraph 'Our database layout'
        house (
            id integer pk,
            name character varying,
            ...more,
            street_id fk
        )

        street (
            id integer pk,
            name text,
            images integer[]
        )

        image (id integer, name text)

        house.street_id -> street.id
        street.images ->> image.id
    %end
    %end xmp

    %dbgraph 'Our database layout'
        house (
            id integer pk,
            name character varying,
            ...more,
            street_id fk
        )

        street (
            id integer pk,
            name text,
            images integer[]
        )

        image (id integer, name text)

        house.street_id -> street.id
        street.images ->> image.id
    %end

## API and options

Dog provides two API functions:

    dog.build_all(mode: str, options: dict) 

builds the documentation. `mode` is `"html"` or `"pdf"`.

    dog.start_server(options: dict)

starts a development server with live reload.

Dog recognizes the following options:

| option            | type        | meaning                                          |
|-------------------|-------------|--------------------------------------------------|
| `assetPatterns`   | `List[str]` | shell patterns for asset files, e.g. `*jpg`      |
| `debug`           | `bool`      | embed debug information in the output            |
| `docPatterns`     | `List[str]` | shell patterns for doc files, e.g `*doc.md`      |
| `excludeRegex`    | `str`       | regex to match file paths that should be ignored |
| `extraAssets`     | `List[str]` | extra asset paths (e.g css, js)                  |
| `htmlSplitLevel`  | `int`       | html split-level, see below                      |
| `includeTemplate` | `str`       | path to a Jump included template, see below      |
| `outputDir`       | `str`       | output directory                                 |
| `pageTemplate`    | `str`       | path to a Jump page template, see below          |
| `rootDirs`        | `List[str]` | source directories                               |
| `serverHost`      | `str`       | host name for the live server                    |
| `serverPort`      | `int`       | port for the live server                         |
| `staticDir`       | `str`       | where to store assets                            |
| `subTitle`        | `str`       | documentation subtitle                           |
| `title`           | `str`       | documentation title                              |
| `verbose`         | `bool`      | enable debug logging                             |
| `webRoot`         | `str`       | prefix all urls with this path                   |

### html split-level

This option indicates how Dog should write html files.

`0` means all documentation will be stored in a single file (`index.html`)

`1` means one file per level-one section:

    /foo  ->  /foo/index.html
    /bar  ->  /bar/index.html

`2` creates separate files for level-one and level-two sections

    /foo      ->  /foo/index.html
    /foo/bob  ->  /foo/bob/index.html
    /foo/fob  ->  /foo/fob/index.html
    /bar      ->  /bar/index.html

and so on.

### include template

This template, if provided, is included in every source file. Can be used to define custom Jump commands.

### page template

A page template is a Jump template which is rendered for each html page. This template gets the following arguments:

| argument      | meaning                                      |
|---------------|----------------------------------------------|
| `breadcrumbs` | array of tuples `(section-url,section-head)` |
| `main`        | main html content for this page              |
| `mainToc`     | main table of contents as html `LI` elements |
| `options`     | options object as defined above              |
| `path`        | html path for this page                      |
| `subTitle`    | documentation subtitle                       |
| `title`       | documentation title                          |

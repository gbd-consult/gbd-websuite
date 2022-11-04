# DOG - the documentation generator.

## Overview

The docs are scattered across the source tree, so that each module or plugin keeps its docs within its folder.
Given a set of root filesystem folders, the Generator collects all matching files (eg `doc/*.md`), combines them together
and generates html or pdf output. Specific filenames and paths do not matter, the final structure of the documentation
is determined solely by sections and their ids ("sids").

The docs are written in Markdown, with some Jump commands.

## Documentation structure

The DOG documentation is a nested hierarchy of section.

A *section* starts with a Markdown header, optionally followed by a colon and a sid.

    # Introduction :/docs/developer/intro

Deeper headings (with more dashes) create a hierarchy.

    # First Top-Level Section
        ## Subsection 1
        ## Subsection 2
    # Second Top-Level Section
        ## Subsection
            ### Sub-sub-section

Heading levels are only used to create hierarchies, the output rendering (h1, h2 etc.) is determined by
the section depth and html split-level.

### Section sids

A section *sid* denotes the section position in the final doc tree and is used to refer to the section from elsewhere.

A sid consists of *components*, separated by a slash. A component name can contain lowercase letter, digits and a dash.

A section can have an absolute sid, which starts with a slash, a relative one, or no sid at all.

The final sid for a section is determined as follows:

- if sid is given and ends with a slash, the title is converted to a component name and appended to the sid
- if sid is absolute, it is taken as is, and the heading hierarchy is ignored
- if the sid is relative, it is appended to the parent section sid
- if there is no parent (topmost section), the sid replaces the last component of the previous sibling sid
- if no sid is given, the section title is appended to the paren

The topmost section in a file must provide a sid and this sid has to be absolute.

Example:

    # First section :/docs/first
        This section has a complete absolute sid.

        ## Alpha :/docs/important
            This sub-section has an absolute sid.
            The parser ignores the hierarchy and makes this section a top-level

        ## Beta :sub
            This sub-section has a relative sid, which is appended to the parent.
            The final sid will be `/docs/first/sub`

        ## Gamma
            This sub-section has no sid.
            The final sid will be `/docs/first/gamma`

    # Second section :second
        This section has a relative sid, which replaces that last component of the sibling.
        The final sid of this section will be `/docs/second`

    # Third section
        This section has no sid at all.
        The final sid of this section will be `/docs/fourth_section`

### Section embedding

You can embed, or physically paste, sections defined elsewhere in the current file.
To do that, use a header with no text and a sid.

    # :/docs/section-to-add

If the sid is relative, it's resolved relatively to the current section id, with respect to . and ..

The sid can contain a `*` in which case all matching section are included, sorted alphabetically by their titles.

### Section linking

A section can be linked to using the standard Markdown notation:

    [](section-id) or [link text](sid)

Relative sids are resolved relatively to the containing section sid.


## Commands

### toc

Creates a table of contents for given section ids. If `depth` is omitted, it defaults to 1.

    @toc depth=N
        sid-1
        sid-2
        ...
    @end

## Options

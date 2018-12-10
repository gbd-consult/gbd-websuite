Working with css
================


CSS support in GBD WebSuite Client is theme-based. Themes are located in ``./src/css/themes``. Each theme must have at least one ``index`` file, which is responsible for the generating of the target css. Our default theme, ``light``, is written in a css-in-js dialect, but you can use whatever you want (sass, less, straight css etc).


css-in-js Syntax
~~~~~~~~~~~~~~~~

Our css-in-js syntax is straight-forward and resembles the actual css. Each ``css.js`` file must export an object of rules: ::

    module.exports = {
        'selector1: {
            property: value,
            property: value,
            ...
        },
        'selector2': {
            ...
        }
        ...
    }

or a function that returns such an object: ::

    module.exports = function(globalObject) {
        return {
            'selector1: {
                property: value,
                property: value,
                ...
            },
            'selector2': {
                ...
            }
            ...
        }
    }

where ``globalObject`` contains global theme options. It's normally defined in the theme's ``index`` file.

Rule objects are plain JS objects: ::

    'someSelector' : {
        backgroundColor: 'red',
        padding: [10, 20, 30, 40],
        marginTop: 50,
    }

Note that use can use unitless values (which will be converted to the default unit, ``px`` ), and arrays in place of space separated values.

The syntax also supports nested selectors, as in ::

    '.someClass' : {
        backgroundColor: 'red',
        '.inside': {
            paddingTop: 30
        },
    }

which results in the following css ::

    .someClass { background-color: red }
    .someClass .inside { padding-top: 30px }

Selectors prefixed with ``&`` are attached to their parent, so this ::

    '.someClass' : {
        backgroundColor: 'red',
        '&.special': {
            paddingTop: 30
        },
    }

produces ::

    .someClass { background-color: red }
    .someClass.special { padding-top: 30px }

Since css-in-js rules are just plain objects, you can use arbitrary javascript in selectors and values, e.g. ::

    [getMySelector()]: {
        backgroundColor: randomColor(),
        padding: TOP_PADDING * 2,
        ...someOtherRule
        // etc...
    }


Default theme
~~~~~~~~~~~~~

The global object in the default theme, called ``v`` for brevity, contains various theme parameters and useful utilities:

Color helpers
-------------

- ``v.COLOR.color-name``

    Returns a `material color <https://www.materialui.co/colors>`_ with that name, like ``v.COLOR.pink300``

- ``v.COLOR.transform-function(base-color, value)``

    Transforms the given color. Available transforms are:

    - opacity
    - lighten
    - brighten
    - darken
    - desaturate
    - saturate

    Example ::

        v.COLOR.opacity('red', 0.5)

Object helpers
--------------

These helpers retun objects, so they must be used with the spread operator ``...``:


- ``v.GOOGLE_SVG(category/name, color)``

    Sets ``backgroundImage`` to a `material icon <https://material.io/tools/icons>`_ from the given category/name. ``color`` defaults to ``v.ICON_COLOR`` if omitted. Example ::


        '.mySelector': {
            ...v.GOOGLE_SVG('image/straighten', 'blue')

- ``v.LOCAL_SVG(filename, color)``

    Sets ``backgroundImage`` to an svg icon placed in ``themes/light/img``. Example ::


        '.mySelector': {
            ...v.LOCAL_SVG('zoom_rectangle', 'cyan')


- ``v.TRANSITION(property)``

    Inserts the default ``transiition`` for the given property::

        '.mySelector': {
            ...v.TRANSITION('left')


- ``v.SHADOW()``

    Inserts the default ``boxShadow``::

        '.mySelector': {
            ...v.SHADOW()

Selector helpers
----------------

These are intended to be used in selectors (using the js key evaluation operator ``[...]``).


- ``v.MEDIA(breakpoint-name)``

    Creates a ``@media screen ...width`` selector for responsive rules. Breakpoint names are similar to those in `bootstrap <https://getbootstrap.com/docs/4.1/layout/overview/#responsive-breakpoints>`_:

    - xsmall
    - small
    - medium
    - large
    - xlarge

    You can also suffix a name with ``+`` (= breakpoint and up) or ``-`` (= breakpoint and down). Examples: ::

        [v.MEDIA('small')]: {

            // only for "small" devices

            'someSelector' {
                width: 300
            }
        }

        [v.MEDIA('medium+')]: {

            // for "medium" and wider devices

            'someSelector' {
                color: 'blue'
            }
        }

        [v.MEDIA('small-')]: {

            // "small" and "xsmall" devices only

            'someSelector' {
                display: 'none'
            }
        }




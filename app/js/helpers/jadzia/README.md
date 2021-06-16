# jadzia

`jadzia` generates css from javascript objects.

###### example:
```javascript
let rule = {
    '#headline': {
        width: 300,
        paddingLeft: 5,

        'div': {
            font: [12, 'Arial'],
        },

        '&.active': {
            color: 'cyan',
            ':hover': {
                color: 'blue'
            }
        }
    }
}

jadzia.css(rule)
```
###### output:
```css
#headline {
    width: 300px;
    padding-left: 5px;
}
#headline div {
    font: 12px Arial;
}
#headline.active {
    color: cyan;
}
#headline.active:hover {
    color: blue;
}
```

`jadzia` is useful if you (like me):

- write lots of complex and highly parametrized CSS
- know javascript and don't want to learn ad-hoc CSS languages
- like to be framework-agnostic and keep CSS where it belongs: in .css files

## input

`jadzia`'s input is a `rule`, which is an object containing css properties.
You can also pass an (arbitrarily nested) array of `rule`s or a function returning a `rule`.
A rule object can also contain other nested rules with their respective selectors.


###### example:
```javascript
let rule = {
    '#headline': {
        // css props
        width: 300,
        paddingLeft: 5,

        // nested rule
        'div': {
            font: [12, 'Arial'],

            // another nested rule
            '.important': {
                fontWeight: 800
            }
        }
    }
}

jadzia.css(rule)
```
###### output:
```css
#headline {
    width: 300px;
    padding-left: 5px;
}
#headline div {
    font: 12px Arial;
}
#headline div .important {
    font-weight: 800;
}
```

### selectors

Selectors are just like css selectors. If a nested selector starts with an `&` or `:`, it's merged with the parent selector.
For comma-separated nested selectors, `jadzia` creates all possible combinations of them.
Selectors that end with an `&` are prepended to the parent.

###### example:
```javascript
let rule = {
    '#headline': {
        width: 300,
        paddingLeft: 5,

        // simple sub-selector
        'div': {
            font: [12, 'Arial'],
        },

        // merge with parent
        '&.active': {
            color: 'cyan'
        },

        // prepend to parent
        '.dark-theme&': {
            color: 'black'
        },

        // create combinations
        'em, strong': {
            opacity: 0.3
        }
    }
}

jadzia.css(rule)
```
###### output:
```css
#headline {
    width: 300px;
    padding-left: 5px;
}
#headline div {
    font: 12px Arial;
}
#headline.active {
    color: cyan;
}
.dark-theme #headline {
    color: black;
}
#headline em {
    opacity: 0.3;
}
#headline strong {
    opacity: 0.3;
}
```

### media selectors

Media selectors are moved to the topmost level and merged.

###### example:
```javascript
let rule = {
    '#headline': {
        '@media screen': {
            '@media (max-width: 800px)': {
                width: 800,
            }
        },
        '@media print': {
            display: 'none'
        }
    },
    article: {
        '@media screen and (max-width: 800px)': {
            width: '100%',
        },
        '@media print': {
            fontSize: 11
        }
    }
}

jadzia.css(rule)
```
###### output:
```css
@media screen and (max-width: 800px) {
    #headline {
        width: 800px;
    }
    article {
        width: 100%;
    }
}
@media print {
    #headline {
        display: none;
    }
    article {
        font-size: 11px;
    }
}
```

### property names

CSS property names can be quoted, or written with an underscore instead of a dash, or in camelCase:

###### example:
```javascript
let rule = {
    '#headline': {
        paddingLeft: 5,
        padding_top: 10,
        'padding-bottom': 20,
        _webkitTransition: 'all 4s ease',
    }
}

jadzia.css(rule)
```
###### output:
```css
#headline {
    padding-left: 5px;
    padding-top: 10px;
    padding-bottom: 20px;
    -webkit-transition: all 4s ease;
}
```

Custom properties (like `--custom-prop`) can be written with two underscores (`__customProp`). Additionally, all unknown CSS properties are treated as custom and quoted. If you need to quote a known property, pass it in the `quote` option. If you need an unknown unquoted property, pass it in the `unquote` option:


###### example:
```javascript
let rule = {
    '#headline': {
        __customProp: 'a',
        unknownProp: 'b',
        newProp: 'c',
        zoom: 3,
    }
}

jadzia.css(rule, {
    quote: ['zoom'], 
    unquote:['new-prop']
})
```
###### output:
```css
#headline {
    --custom-prop: a;
    --unknown-prop: b;
    new-prop: c;
    --zoom: 3;
}
```



### property values

A property value can be:

- a string, which is taken as is
- an _empty_ string, which will appear in css as `''` (useful for `content` props)
- a number, to which the default unit will be added if required
- an array, which will appear space-joined
- `null`, in which case the property will be removed (useful when extending base rules)
- a function returning one of the above

###### example:
```javascript
const minMargin = 5;

const baseBlock = {
    display: 'block',
    opacity: 0.3,
    backgroundColor: 'cyan',
}

let rule = {
    '#headline': {
        paddingLeft: 5,
        color: 'cyan',
        border: [1, 'dotted', 'white'],

        margin: () => [1, 3].map(x => x + minMargin),
        content: '',

        ...baseBlock,
        backgroundColor: null,
    }
}

jadzia.css(rule)
```
###### output:
```css
#headline {
    padding-left: 5px;
    color: cyan;
    border: 1px dotted white;
    margin: 6px 8px;
    content: '';
    display: block;
    opacity: 0.3;
}
```

## API

```
jadzia.css(rules, options)
```
takes `rules` and returns formatted CSS.

```
jadzia.object(rules, options)
```
takes `rules` and returns a normalized CSS object.

```
jadzia.format(object, options)
```
formats a normalized object into CSS.


###### example:
```javascript
let rule = {
    '#headline': {
        width: 300,
        paddingLeft: 5,

        'div': {
            font: [12, 'Arial'],
        }
    }
}

JSON.stringify(jadzia.object(rule), null, 4)
```
###### output:
```javascript
{
    "#headline": {
        "width": "300px",
        "padding-left": "5px"
    },
    "#headline div": {
        "font": "12px Arial"
    }
}
```


The options are:

option|    |default
------|----|----
`quote` | list of properties that should be preceded with two dashes | `[]`
`unquote` | list of properties that should not be preceded with two dashes | `[]`
`indent` | indentation for the generated CSS | `4`
`sort` | sort selectors and property values | `false`
`unit` | default unit for numeric values | `px`

###### example:
```javascript
let rule = {
    '#headline': {
        zIndex: 3,
        padding: 4,
        zoom: 5,
    }
}

jadzia.css(rule, {
    unit: 'em',
    indent: 2,
    sort: true,
    quote: ['zoom'],
})
```
###### output:
```css
#headline {
  --zoom: 5;
  padding: 4em;
  z-index: 3;
}
```

`options` are passed to selector and property functions. You can put your own values there for CSS parametrization:


###### example:
```javascript
let rule = options => ({
    '#headline': {
        width: 300,
        color: options.textColor,
        border: [1, 'dotted', options.borderColor],
    }
});

jadzia.css(rule, {
    textColor: '#cc0000',
    borderColor: 'cyan'
})
```
###### output:
```css
#headline {
    width: 300px;
    color: #cc0000;
    border: 1px dotted cyan;
}
```

## info

(c) 2019 Georg Barikin (https://github.com/gebrkn). MIT license.



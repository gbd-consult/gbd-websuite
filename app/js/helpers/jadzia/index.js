//

module.exports.css = function css(rules, options) {
    options = _defopts(options);
    return _format(_parse(rules, options), options);
};

module.exports.object = function object(rules, options) {
    return _parse(rules, _defopts(options));
};

module.exports.format = function format(obj, options) {
    return _format(obj, _defopts(options));
};

//

const _defaults = {
    unit: 'px',
    sort: false,
    indent: 4,
    quote: null,
    unquote: null,
};

function _defopts(options) {
    return Object.assign({}, _defaults, options || {});
}

function _parse(rules, options) {

    // first, flatten the nested structure, e.g. [ {a:{b:{c:val}}} ]....
    // into a list {selector, prop, value}, e.g. {selector: [a,b], prop: c, value: val}

    let flat = _flatten(rules, options);

    // move media selectors forward and merge others
    // (eg ["foo", "&.bar", "baz", "@media blah"] => ["@media blah", "foo.bar baz"])

    _each(flat, item => item.selector = _parseSelector(item.selector));

    // convert the flat structure back to a nested object

    let obj = _unflatten(flat, options);

    // sort by selector/property name

    if (options.sort) {
        obj = _sort(obj);
    }

    return obj;
}

function _flatten(rules, options) {

    let res = [];

    let walk = (obj, keys) => {
        while (_type(obj) === T_FUNCTION) {
            obj = obj(options);
        }

        if (_type(obj) === T_NULL) {
            return;
        }

        if (_type(obj) === T_ARRAY) {
            _each(obj, x => walk(x, keys));
            return;
        }

        if (_type(obj) === T_OBJECT) {
            _each(obj, (val, key) => {
                _each(_split(key), k => walk(val, keys.concat(k)));
            });
            return;
        }

        res.push({
            selector: keys.slice(0, -1),
            prop: keys[keys.length - 1],
            value: obj
        })
    };

    walk(rules, []);
    return res;
}

function _parseSelector(sel) {
    let r = {
        at: [],
        media: [],
        prepend: [],
        rest: []
    };

    function merge(sel) {
        sel = sel.join('\x01');
        sel = sel.replace(/\x01&\s*/g, '');
        sel = sel.replace(/\x01:\s*/g, ':');
        sel = sel.replace(/\x01/g, ' ');
        return sel;
    }

    _each(sel, s => {
        if (_startsWith(s, '@media')) {
            r.media.push(_trim(s.replace(/^@media/, '')));
            return;
        }

        if (_startsWith(s, '@')) {
            // remove everything before an at-rule
            r.rest = [];
            r.at.push(s);
            return;
        }

        if (_endsWith(s, '&')) {
            if (r.rest.length > 0) {
                let last = r.rest.pop();
                r.rest.push(_trim(s.slice(0, -1)));
                r.rest.push(last);
            }
            return;
        }

        r.rest.push(s);
    });

    let res = [];

    if (r.media.length > 0) {
        res.push('@media ' + r.media.join(' and '));
    }

    if (r.at.length > 0) {
        res = res.concat(r.at);
    }

    res.push(_trim(r.prepend.join(' ') + ' ' + merge(r.rest)));

    return res.filter(Boolean);
}

function _sort(obj) {
    let cmp = (a, b) => (a > b) - (a < b);

    function weight(s) {
        if (!s) return '00';

        if (s[0] === '*') return '10';
        if (s[0] === '#') return '20';
        if (s[0] === '.') return '30';
        if (s[0] === '[') return '40';
        if (s[0] === '@') return '90';

        return '15';
    }

    function byWeight(a, b) {
        return cmp(weight(a) + a, weight(b) + b);
    }

    let r = {};

    _each(Object.keys(obj).sort(byWeight), k => {
        let v = obj[k];
        if (_type(v) === T_OBJECT)
            v = _sort(v);
        r[k] = v;
    });

    return r;
}

function _unflatten(flat, options) {
    let res = {};

    _each(flat, c => {

        let cur = res;

        _each(c.selector, sel => {
            switch (_type(cur[sel])) {
                case T_NULL:
                    cur[sel] = {};
                    cur = cur[sel];
                    return;
                case T_OBJECT:
                    cur = cur[sel];
                    return;
                default:
                    throw new Error('nesting error');
            }
        });

        let name = _propName(c.prop, options);
        let value = _propValue(c.value, name, options);

        if (_type(value) !== T_NULL)
            cur[name] = value;
    });

    return res;
}

function _format(obj, options) {
    let lines = [];
    let indent = Number(options.indent) || 0;

    function write(val, key, level) {
        let ind = _repeat(' ', level * indent);

        if (_type(val) !== T_OBJECT) {
            lines.push(ind + key + ': ' + val + ';');
            return;
        }
        lines.push(ind + key + ' {');
        _each(val, (v, k) => write(v, k, level + 1));
        lines.push(ind + '}');
    }

    _each(obj, (v, k) => write(v, k, 0));
    return lines.join('\n');
}

// couldn't find a complete list of css props and units, so this is manually compiled from
// https://svn.webkit.org/repository/webkit/trunk/Source/WebCore/css/CSSProperties.json (and spec links therein)
// and https://github.com/rofrischmann/unitless-css-property/blob/master/modules/index.js

const _allProps = {
    "align-content": 0,
    "align-items": 0,
    "align-self": 0,
    "alignment-baseline": 0,
    "all": 0,
    "alt": 0,
    "animation": 0,
    "animation-delay": 0,
    "animation-direction": 0,
    "animation-duration": 0,
    "animation-fill-mode": 0,
    "animation-iteration-count": 1,
    "animation-name": 0,
    "animation-play-state": 0,
    "animation-timing-function": 0,
    "background": 0,
    "background-attachment": 0,
    "background-blend-mode": 0,
    "background-clip": 0,
    "background-color": 0,
    "background-image": 0,
    "background-origin": 0,
    "background-position": 0,
    "background-repeat": 0,
    "background-size": 0,
    "baseline-shift": 0,
    "block-size": 0,
    "border": 0,
    "border-block": 0,
    "border-block-color": 0,
    "border-block-end": 0,
    "border-block-end-color": 0,
    "border-block-end-style": 0,
    "border-block-end-width": 0,
    "border-block-start": 0,
    "border-block-start-color": 0,
    "border-block-start-style": 0,
    "border-block-start-width": 0,
    "border-block-style": 0,
    "border-block-width": 0,
    "border-bottom": 0,
    "border-bottom-color": 0,
    "border-bottom-left-radius": 0,
    "border-bottom-right-radius": 0,
    "border-bottom-style": 0,
    "border-bottom-width": 0,
    "border-boundary": 0,
    "border-collapse": 0,
    "border-color": 0,
    "border-image": 0,
    "border-image-outset": 1,
    "border-image-repeat": 0,
    "border-image-slice": 1,
    "border-image-source": 0,
    "border-image-width": 1,
    "border-inline": 0,
    "border-inline-color": 0,
    "border-inline-end": 0,
    "border-inline-end-color": 0,
    "border-inline-end-style": 0,
    "border-inline-end-width": 0,
    "border-inline-start": 0,
    "border-inline-start-color": 0,
    "border-inline-start-style": 0,
    "border-inline-start-width": 0,
    "border-inline-style": 0,
    "border-inline-width": 0,
    "border-left": 0,
    "border-left-color": 0,
    "border-left-style": 0,
    "border-left-width": 0,
    "border-radius": 0,
    "border-right": 0,
    "border-right-color": 0,
    "border-right-style": 0,
    "border-right-width": 0,
    "border-spacing": 0,
    "border-style": 0,
    "border-top": 0,
    "border-top-color": 0,
    "border-top-left-radius": 0,
    "border-top-right-radius": 0,
    "border-top-style": 0,
    "border-top-width": 0,
    "border-width": 0,
    "bottom": 0,
    "box-shadow": 0,
    "box-sizing": 0,
    "break-after": 0,
    "break-before": 0,
    "break-inside": 0,
    "buffered-rendering": 0,
    "caption-side": 0,
    "caret-color": 0,
    "clear": 0,
    "clip": 0,
    "clip-path": 0,
    "clip-rule": 0,
    "color": 0,
    "color-interpolation": 0,
    "color-interpolation-filters": 0,
    "color-rendering": 0,
    "column-count": 1,
    "column-fill": 0,
    "column-gap": 0,
    "column-rule": 0,
    "column-rule-color": 0,
    "column-rule-style": 0,
    "column-rule-width": 0,
    "column-span": 0,
    "column-width": 0,
    "columns": 0,
    "content": 0,
    "counter-increment": 0,
    "counter-reset": 0,
    "cursor": 0,
    "cx": 0,
    "cy": 0,
    "direction": 0,
    "display": 0,
    "dominant-baseline": 0,
    "empty-cells": 0,
    "fill": 0,
    "fill-color": 0,
    "fill-image": 0,
    "fill-opacity": 1,
    "fill-origin": 0,
    "fill-position": 0,
    "fill-rule": 0,
    "filter": 0,
    "flex": 1,
    "flex-basis": 0,
    "flex-direction": 0,
    "flex-flow": 0,
    "flex-grow": 1,
    "flex-shrink": 1,
    "flex-wrap": 0,
    "float": 0,
    "flood-color": 0,
    "flood-opacity": 1,
    "font": 0,
    "font-display": 0,
    "font-family": 0,
    "font-feature-settings": 0,
    "font-optical-sizing": 0,
    "font-size": 0,
    "font-stretch": 0,
    "font-style": 0,
    "font-synthesis": 0,
    "font-variant": 0,
    "font-variant-alternates": 0,
    "font-variant-caps": 0,
    "font-variant-east-asian": 0,
    "font-variant-ligatures": 0,
    "font-variant-numeric": 0,
    "font-variation-settings": 0,
    "font-weight": 1,
    "gap": 0,
    "glyph-orientation-horizontal": 0,
    "glyph-orientation-vertical": 0,
    "grid": 0,
    "grid-area": 0,
    "grid-auto-columns": 0,
    "grid-auto-flow": 0,
    "grid-auto-rows": 0,
    "grid-column": 1,
    "grid-column-end": 0,
    "grid-column-start": 0,
    "grid-row": 1,
    "grid-row-end": 0,
    "grid-row-start": 0,
    "grid-template": 0,
    "grid-template-areas": 0,
    "grid-template-columns": 0,
    "grid-template-rows": 0,
    "hanging-punctuation": 0,
    "height": 0,
    "image-orientation": 0,
    "image-rendering": 0,
    "inline-size": 0,
    "inset": 0,
    "inset-block": 0,
    "inset-block-end": 0,
    "inset-block-start": 0,
    "inset-inline": 0,
    "inset-inline-end": 0,
    "inset-inline-start": 0,
    "isolation": 0,
    "justify-content": 0,
    "justify-items": 0,
    "justify-self": 0,
    "kerning": 0,
    "left": 0,
    "letter-spacing": 0,
    "lighting-color": 0,
    "line-break": 0,
    "line-height": 1,
    "list-style": 0,
    "list-style-image": 0,
    "list-style-position": 0,
    "list-style-type": 0,
    "margin": 0,
    "margin-block": 0,
    "margin-block-end": 0,
    "margin-block-start": 0,
    "margin-bottom": 0,
    "margin-inline": 0,
    "margin-inline-end": 0,
    "margin-inline-start": 0,
    "margin-left": 0,
    "margin-right": 0,
    "margin-top": 0,
    "marker": 0,
    "marker-end": 0,
    "marker-mid": 0,
    "marker-start": 0,
    "mask": 0,
    "mask-type": 0,
    "max-block-size": 0,
    "max-height": 0,
    "max-inline-size": 0,
    "max-width": 0,
    "max-zoom": 1,
    "min-block-size": 0,
    "min-height": 0,
    "min-inline-size": 0,
    "min-width": 0,
    "min-zoom": 1,
    "mix-blend-mode": 0,
    "object-fit": 0,
    "object-position": 0,
    "opacity": 1,
    "order": 1,
    "orientation": 0,
    "orphans": 1,
    "outline": 0,
    "outline-color": 0,
    "outline-offset": 0,
    "outline-style": 0,
    "outline-width": 0,
    "overflow": 0,
    "overflow-wrap": 0,
    "overflow-x": 0,
    "overflow-y": 0,
    "padding": 0,
    "padding-block": 0,
    "padding-block-end": 0,
    "padding-block-start": 0,
    "padding-bottom": 0,
    "padding-inline": 0,
    "padding-inline-end": 0,
    "padding-inline-start": 0,
    "padding-left": 0,
    "padding-right": 0,
    "padding-top": 0,
    "page": 0,
    "page-break-after": 0,
    "page-break-before": 0,
    "page-break-inside": 0,
    "paint-order": 0,
    "perspective": 0,
    "perspective-origin": 0,
    "perspective-origin-x": 0,
    "perspective-origin-y": 0,
    "place-content": 0,
    "place-items": 0,
    "place-self": 0,
    "pointer-events": 0,
    "position": 0,
    "quotes": 0,
    "r": 0,
    "resize": 0,
    "right": 0,
    "row-gap": 0,
    "rx": 0,
    "ry": 0,
    "scroll-padding": 0,
    "scroll-padding-bottom": 0,
    "scroll-padding-left": 0,
    "scroll-padding-right": 0,
    "scroll-padding-top": 0,
    "scroll-snap-align": 0,
    "scroll-snap-margin": 0,
    "scroll-snap-margin-bottom": 0,
    "scroll-snap-margin-left": 0,
    "scroll-snap-margin-right": 0,
    "scroll-snap-margin-top": 0,
    "scroll-snap-type": 0,
    "shape-image-threshold": 1,
    "shape-margin": 0,
    "shape-outside": 0,
    "shape-rendering": 0,
    "size": 0,
    "speak-as": 0,
    "src": 0,
    "stop-color": 0,
    "stop-opacity": 1,
    "stroke": 0,
    "stroke-color": 0,
    "stroke-dasharray": 1,
    "stroke-dashoffset": 1,
    "stroke-linecap": 0,
    "stroke-linejoin": 0,
    "stroke-miterlimit": 1,
    "stroke-opacity": 1,
    "stroke-width": 0,
    "tab-size": 1,
    "table-layout": 0,
    "text-align": 0,
    "text-anchor": 0,
    "text-decoration": 0,
    "text-indent": 0,
    "text-line-through": 0,
    "text-line-through-color": 0,
    "text-line-through-mode": 0,
    "text-line-through-style": 0,
    "text-line-through-width": 1,
    "text-overflow": 0,
    "text-overline": 0,
    "text-overline-color": 0,
    "text-overline-mode": 0,
    "text-overline-style": 0,
    "text-overline-width": 1,
    "text-rendering": 0,
    "text-shadow": 0,
    "text-transform": 0,
    "text-underline": 0,
    "text-underline-color": 0,
    "text-underline-mode": 0,
    "text-underline-style": 0,
    "text-underline-width": 1,
    "top": 0,
    "touch-action": 0,
    "transform": 0,
    "transform-box": 0,
    "transform-origin": 0,
    "transform-origin-x": 0,
    "transform-origin-y": 0,
    "transform-origin-z": 0,
    "transform-style": 0,
    "transition": 0,
    "transition-delay": 0,
    "transition-duration": 0,
    "transition-property": 0,
    "transition-timing-function": 0,
    "unicode-bidi": 0,
    "unicode-range": 0,
    "user-zoom": 0,
    "vector-effect": 0,
    "vertical-align": 0,
    "visibility": 0,
    "white-space": 0,
    "widows": 1,
    "width": 0,
    "will-change": 0,
    "word-break": 0,
    "word-spacing": 0,
    "word-wrap": 0,
    "writing-mode": 0,
    "x": 0,
    "y": 0,
    "z-index": 1,
    "zoom": 1
};

const _vendors = ['webkit', 'moz', 'ms', 'o'];

function _propToCSS(key) {
    return String(key)
        .replace(/[A-Z]/g, '-$&')
        .replace(/_/g, '-')
        .toLowerCase();
}

function _isProp(name) {
    if (!_isProp.cache) {
        _isProp.cache = {};
        _each(_allProps, (isUnitless, prop) => {
            _isProp.cache[prop] = isUnitless;
            _each(_vendors, v => _isProp.cache['-' + v + '-' + prop] = isUnitless);
        })
    }
    return name in _isProp.cache;
}

function _isUnitless(name) {
    return !_isProp(name) || _isProp.cache[name];
}

function _propName(key, options) {
    let quote = '--';
    let name = _propToCSS(key);

    if (_startsWith(name, quote))
        return name;

    if (options.quote && options.quote.includes(name))
        return quote + name;

    if (_isProp(name))
        return name;

    if (options.unquote && options.unquote.includes(name))
        return name;

    return quote + name;
}

function _propValue(val, name, options) {
    let t = _type(val);

    if (t === T_NULL)
        return null;

    if (t === T_SIMPLEARRAY)
        val = _trim(val.map(v => _propValue(v, name, options)).join(' '));

    if (val === '')
        return "''";

    if (t === T_NUMBER && val !== 0 && !_isUnitless(name))
        return String(val) + options.unit;

    return String(val);
}

function _each(obj, fn) {
    let t = _type(obj);

    if (t === T_ARRAY || t === T_SIMPLEARRAY) {
        obj.forEach(fn);
    } else if (t === T_OBJECT) {
        Object.keys(obj).forEach(k => fn(obj[k], k));
    }
}

function _split(x) {
    x = String(x);

    // for empty child selectors
    if (x.length === 0) {
        return [''];
    }

    return x.split(',').map(_trim).filter(Boolean);
}

function _trim(x) {
    return String(x).trim();
}

function _startsWith(x, y) {
    return String(x).indexOf(y) === 0;
}

function _endsWith(x, y) {
    x = String(x);
    y = String(y);
    return x.slice(x.length - y.length) === y;

}

function _repeat(s, n) {
    let t = '';
    while (n--) {
        t += s;
    }
    return t;
}

const T_ARRAY = 1;
const T_BOOLEAN = 2;
const T_FUNCTION = 3;
const T_NULL = 4;
const T_NUMBER = 5;
const T_OBJECT = 6;
const T_SIMPLEARRAY = 7;
const T_STRING = 8;
const T_SYMBOL = 9;

function _type(v) {
    function isprim(v) {
        let t = typeof v;
        return t === 'number' || t === 'bigint' || t === 'string' || t === 'boolean';
    }

    let t = typeof v;

    if (v === null || t === 'undefined') {
        return T_NULL;
    }

    if (Array.isArray(v)) {
        if (v.length === 0) {
            return T_NULL;
        }
        if (v.every(isprim)) {
            return T_SIMPLEARRAY;
        }
        return T_ARRAY;
    }

    switch (t) {
        case 'string':
            return T_STRING;
        case 'object':
            return (Object.keys(v).length === 0) ? T_NULL : T_OBJECT;
        case 'boolean':
            return T_BOOLEAN;
        case 'bigint':
        case 'number':
            return T_NUMBER;
        case 'function':
            return T_FUNCTION;
        case 'symbol':
            return T_SYMBOL;
    }
}

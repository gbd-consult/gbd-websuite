// compile js declarations into css styles

// rules is an array of
// - objects (css rules)
// - functions (expecting options as an argument)
// - arrays of such things, recursively
//
// options are {unit, sort, customProps + any vars to be passed to functions}

module.exports.css = function toCss(rules, options) {
    options = Object.assign({}, _defaults, options || {});
    let obj = _parse(rules, options);
    let css = _toCSS(obj, options);
    if (options.indent) {
        css = _indent(css, options.indent);
    }
    return css.join('\n');


};

const _defaults = {
    unit: 'px',
    sort: true,
    indent: 4,
    customProps: null,
};


function _parse(rules, options) {

    // first, flatten the nested structure, e.g. [ {a:{b:{c:val}}} ]....
    // into a list {selector, prop, value}, e.g. {selector: [a,b], prop: c, value: val}

    let flat = _flatten(rules, options);

    // move media selectors forward and merge others
    // (eg ["foo", "&.bar", "baz", "@media blah"] => ["@media blah", "foo.bar baz"])

    _each(flat, item => item.selector = _parseSelector(item.selector));

    // sort by selector/property name

    if (options.sort) {
        flat = _sort(flat);
    }

    // convert the flat structure back to nested objects

    return _unflatten(flat);
}


function _flatten(rules, options) {

    let res = [];

    let walk = (obj, keys) => {
        while (_type(obj) === 'function') {
            obj = obj(options);
        }

        if (_type(obj) === 'null') {
            return;
        }

        if (_type(obj) === 'array') {
            _each(obj, x => walk(x, keys));
            return;
        }

        if (_type(obj) === 'object') {
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


function _sort(flat) {
    let cmp = (a, b) => (a > b) - (a < b);

    let key = item => [].concat(item.selector.map(weight), item.prop, String(item.value));

    function weight(s) {
        if (!s) return '00';

        if (s[0] === '*') return '10' + s;
        if (s[0] === '#') return '20' + s;
        if (s[0] === '.') return '30' + s;
        if (s[0] === '[') return '40' + s;
        if (s[0] === '@') return '90' + s;

        return '15' + s;
    }

    function cmpArr(a, b) {
        while (a.length > 0 && b.length > 0) {
            let c = cmp(a.shift(), b.shift());
            if (c !== 0)
                return c;
        }
        return a.length - b.length;
    }

    return flat.sort((a, b) => cmpArr(key(a), key(b)));
}

function _unflatten(flat) {
    let res = {};

    _each(flat, item => {

        let cur = res;

        _each(item.selector, sel => {
            switch (_type(cur[sel])) {
                case 'null':
                    cur[sel] = {};
                    cur = cur[sel];
                    return;
                case 'object':
                    cur = cur[sel];
                    return;
                default:
                    throw new Error('nesting error');
            }
        });

        cur[item.prop] = item.value;
    });

    return res;
}


function _toCSS(obj, options) {
    let lines = [];

    function selector(s) {
        return _trim(s);
    }

    function write(val, key) {
        if (_type(val) === 'object') {
            lines.push(selector(key) + ' {');
            _each(val, write);
            lines.push('}');
            return;
        }

        let name = _propName(key, options);
        let value = _propValue(val, name, options);
        if (_type(value) !== 'null')
            lines.push(name + ': ' + value + ';');
    }

    _each(obj, write);
    return lines;
}

// from https://github.com/rofrischmann/unitless-css-property/blob/master/modules/index.js

const _unitless = [
    'animationIterationCount',
    'borderImageOutset',
    'borderImageSlice',
    'borderImageWidth',
    'boxFlex',
    'boxFlexGroup',
    'boxOrdinalGroup',
    'columnCount',
    'fillOpacity',
    'flex',
    'flexGrow',
    'flexNegative',
    'flexOrder',
    'flexPositive',
    'flexShrink',
    'floodOpacity',
    'fontWeight',
    'gridColumn',
    'gridRow',
    'lineClamp',
    'lineHeight',
    'opacity',
    'order',
    'orphans',
    'stopOpacity',
    'strokeDasharray',
    'strokeDashoffset',
    'strokeMiterlimit',
    'strokeOpacity',
    'strokeWidth',
    'tabSize',
    'widows',
    'zIndex',
    'zoom',
];

const _vendors = ['webkit', 'moz', 'ms', 'o'];

function _propToCSS(key) {
    return String(key)
        .replace(/[A-Z]/g, '-$&')
        .replace(/_/g, '-')
        .toLowerCase();
}


function _isUnitless(name) {
    if (!_unitless.cache) {
        _unitless.cache = {};
        _each(_unitless, key => {
            let n = _propToCSS(key);
            _unitless.cache[n] = 1;
            _each(_vendors, v => _unitless.cache['-' + v + '-' + n] = 1);
        })
    }

    return (name in _unitless.cache);
}

function _propName(key, options) {
    let name = _propToCSS(key);
    if (options.customProps && options.customProps.includes(name)) {
        name = '--' + name;
    }
    return name;
}

function _propValue(val, name, options) {
    if (_type(val) === 'null')
        return null;

    if (_type(val) === 'simpleArray')
        val = _trim(val.map(v => _propValue(v, name, options)).join(' '));

    if (val === '')
        return "''";

    if (_type(val) === 'number' && val !== 0 && !_isUnitless(name))
        return String(val) + options.unit;

    return val;
}


function _indent(lines, size) {
    let level = 0;

    return lines.map(s => {
        if (_endsWith(s, '}'))
            level--;
        let ii = _repeat(' ', level * size);
        if (_endsWith(s, '{'))
            level++;
        return ii + s;
    });
}


function _type(v) {
    let t = typeof v;

    if (v === null || t === 'undefined') {
        return 'null';
    }

    if (Array.isArray(v)) {
        if (v.length === 0) {
            return 'null';
        }
        if (v.every(_isPrimitive)) {
            return 'simpleArray'
        }
        return 'array';
    }

    if (t === 'object') {
        if (Object.keys(v).length === 0) {
            return 'null';
        }
        return 'object';
    }

    return t;
}

function _isPrimitive(v) {
    let t = typeof v;
    return t === 'number' || t === 'string' || t === 'boolean'
}

function _each(obj, fn) {
    let t = _type(obj);

    if (t === 'array' || t === 'simpleArray') {
        obj.forEach(fn);
    } else if (t === 'object') {
        Object.keys(obj).forEach(k => fn(obj[k], k));
    }
}

function _split(x) {
    x = String(x);

    // for empty child selectors
    if(x.length === 0) {
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